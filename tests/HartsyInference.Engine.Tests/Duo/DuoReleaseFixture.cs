using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Duo;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.ModelAssets.Duo;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tests.Duo;

namespace HartsyInference.Engine.Tests.Duo;

/// <summary>A tiny synthetic Duo release folder (2-block backbone in BF16 with per-head qkv rows, a causal or distilled LoRA in a
/// distributed checkpoint, placeholder codec files) and a two-block run folder for that stage, both written to a temporary directory. The codecs are fake decoders
/// whose pixels and samples depend on their input window.</summary>
internal sealed class DuoReleaseFixture : IDisposable
{
    public const int Heads = 2, HeadDim = 96, Inner = Heads * HeadDim, Hidden = 8, Ffn = 16, TextDim = 5, Layers = 2, Rank = 2;
    public const int Size = 64, Blocks = 2, Frames = 34, Steps = 2;
    public const int SharedRows = 3, BodyRows = 2;
    private const int VideoIn = 96, AudioIn = 32, RefRows = 4, VideoRows = 20, AudioRows = 58;

    public DuoReleaseFixture(bool distilled = false)
    {
        Root = Path.Combine(Path.GetTempPath(), $"duo-recipe-{Guid.NewGuid():N}");
        Release = Path.Combine(Root, "release");
        RunFolder = Path.Combine(Root, "run");
        WriteBackbone(Backbone);
        WriteLora(Dcp, distilled ? "models.generator." : "models.student.");
        Touch(Path.Combine(Release, "codec", "video_vae", "model.safetensors"));
        Touch(Path.Combine(Release, "codec", "audio_vae", "diffusion_pytorch_model.safetensors"));
        File.WriteAllText(Path.Combine(Release, "realization", "complete.json"),
            $$"""{ "stage": "{{(distilled ? "dmd" : "causal")}}" }""");
        WriteRun(RunFolder, distilled);
    }

    public string Root { get; }

    public string Release { get; }

    public string RunFolder { get; }

    public string Backbone => Path.Combine(Release, "realization", "backbone");

    public string Dcp => Path.Combine(Release, "realization", "dcp");

    /// <summary>Fake codecs: video pixels follow the window's mean, and audio sample <c>s</c> of channel <c>c</c> of a window is
    /// <c>mean + s·1e-6 + c·0.25</c>, so a soundtrack shows which samples of which block it kept.</summary>
    public static DuoCodecs Codecs() => new(
        backend => new DuoVideoStreamDecoder(backend, Video),
        backend => new DuoAudioStreamDecoder(backend, Audio, AudioIn, DuoCodecSpec.AudioHop, DuoCodecSpec.Get("duo_5_17").AudioLatents,
            lookaheadSteps: 0, contextSteps: 0),
        static () => []);

    public static void WriteLora(string folder, string prefix)
    {
        Dictionary<string, DcpFixture.Tensor1> tensors = [];
        (string Module, int In, int Out)[] modules =
        [
            ("transformer_blocks.0.attn.to_q", Hidden, Inner), ("transformer_blocks.0.attn.to_k", Hidden, Inner),
            ("transformer_blocks.1.attn.to_v", Hidden, Inner), ("transformer_blocks.0.ff.net.0.proj", Hidden, 2 * Ffn),
            ("transformer_blocks.1.attn.to_out.0", Inner, Hidden), ("proj_out", Hidden, VideoIn),
        ];
        int seed = 200;
        foreach ((string module, int inDim, int outDim) in modules)
        {
            tensors[$"{prefix}{module}.lora_A.default.weight"] = DcpFixture.F32([Rank, inDim], Values(seed++, Rank * inDim, 0.5f));
            tensors[$"{prefix}{module}.lora_B.default.weight"] = DcpFixture.F32([outDim, Rank], Values(seed++, outDim * Rank, 0.5f));
        }
        DcpFixture.Write(folder, tensors);
    }

    public void Dispose()
    {
        if (Directory.Exists(Root))
        {
            Directory.Delete(Root, recursive: true);
        }
    }

    private static Tensor Video(Tensor window)
    {
        int frames = DuoVideoDecodePlan.RawFramesFor((int)window.Shape[2]);
        int height = (int)window.Shape[3] * 16, width = (int)window.Shape[4] * 16;
        float mean = Mean(window);
        Tensor output = new(new TensorShape(new long[] { 1, 3, frames, height, width }), DType.F32);
        Span<float> o = output.AsSpan<float>();
        for (int i = 0; i < o.Length; i++)
        {
            o[i] = MathF.Tanh((mean * 8f) + (i % 997 * 0.001f));
        }
        return output;
    }

    private static Tensor Audio(Tensor window)
    {
        int samples = (int)window.Shape[3] * DuoCodecSpec.AudioHop;
        float mean = Mean(window);
        Tensor output = new(new TensorShape(new long[] { 1, 2, samples }), DType.F32);
        Span<float> o = output.AsSpan<float>();
        for (int channel = 0; channel < 2; channel++)
        {
            for (int s = 0; s < samples; s++)
            {
                o[(channel * samples) + s] = mean + (s * 1e-6f) + (channel * 0.25f);
            }
        }
        return output;
    }

    private static float Mean(Tensor window)
    {
        double sum = 0;
        foreach (float value in window.AsSpan<float>())
        {
            sum += value;
        }
        return (float)(sum / window.ElementCount);
    }

    private static void Touch(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, []);
    }

    private static void WriteBackbone(string folder)
    {
        Directory.CreateDirectory(folder);
        List<SafeTensorsStreamWriter.Entry> shared =
        [
            F32("video_patch_proj.weight", [Hidden, VideoIn], Values(10, Hidden * VideoIn)),
            F32("audio_patch_proj.weight", [Hidden, AudioIn], Values(11, Hidden * AudioIn)),
            F32("condition_proj.weight", [Hidden, TextDim], Values(12, Hidden * TextDim)),
            F32("time_embedder.proj_in.weight", [8, 256], Values(13, 8 * 256, 0.2f)),
            F32("time_embedder.proj_out.weight", [6, 8], Values(14, 6 * 8)),
            F32("rope.inv_freq", [16], Ones(16)),
            Bf16("token_refiner.blocks.0.attn.qkv_proj.weight", [3 * Inner, Hidden], Values(15, 3 * Inner * Hidden)),
            Bf16("token_refiner.blocks.0.attn.out_proj.weight", [Hidden, Inner], Values(16, Hidden * Inner, 0.2f)),
            Bf16("token_refiner.blocks.0.mlp.fc1.weight", [2 * Ffn, Hidden], Values(17, 2 * Ffn * Hidden)),
            Bf16("token_refiner.blocks.0.mlp.fc2.weight", [Hidden, Ffn], Values(18, Hidden * Ffn)),
            F32("token_refiner.blocks.0.attn.q_norm.weight", [HeadDim], Ones(HeadDim)),
            F32("token_refiner.blocks.0.attn.k_norm.weight", [HeadDim], Ones(HeadDim)),
            F32("token_refiner.blocks.0.norm1.weight", [Hidden], Ones(Hidden)),
            F32("token_refiner.blocks.0.norm2.weight", [Hidden], Ones(Hidden)),
            F32("token_refiner.final_norm.weight", [Hidden], Ones(Hidden)),
            F32("final_layer.norm.weight", [Hidden], Ones(Hidden)),
            Bf16("final_layer.adaln_proj.linear.weight", [2 * Hidden, 6], Values(24, 2 * Hidden * 6, 0.5f)),
            F32("final_layer.video_out.weight", [VideoIn, Hidden], Values(25, VideoIn * Hidden)),
            F32("final_layer.audio_out.weight", [AudioIn, Hidden], Values(26, AudioIn * Hidden)),
        ];
        List<SafeTensorsStreamWriter.Entry> blocks = [];
        for (int layer = 0; layer < Layers; layer++)
        {
            int seed = 100 * (layer + 1);
            string p = $"blocks.{layer}";
            blocks.Add(F32($"{p}.norm1.weight", [Hidden], Ones(Hidden)));
            blocks.Add(F32($"{p}.norm2.weight", [Hidden], Ones(Hidden)));
            blocks.Add(Bf16($"{p}.adaln_proj.linear.weight", [18 * Hidden, 6], Values(seed, 18 * Hidden * 6, 0.5f)));
            blocks.Add(Bf16($"{p}.attn.qkv_proj.weight", [3 * Inner, Hidden], Values(seed + 1, 3 * Inner * Hidden)));
            blocks.Add(F32($"{p}.attn.q_norm.weight", [HeadDim], Ones(HeadDim)));
            blocks.Add(F32($"{p}.attn.k_norm.weight", [HeadDim], Ones(HeadDim)));
            blocks.Add(Bf16($"{p}.attn.out_proj.weight", [Hidden, Inner], Values(seed + 2, Hidden * Inner, 0.2f)));
            blocks.Add(Bf16($"{p}.mlp.fc1.weight", [2 * Ffn, Hidden], Values(seed + 3, 2 * Ffn * Hidden)));
            blocks.Add(Bf16($"{p}.mlp.fc2.weight", [Hidden, Ffn], Values(seed + 4, Hidden * Ffn)));
        }
        SafeTensorsStreamWriter.Write(Path.Combine(folder, "model-00001-of-00002.safetensors"), shared);
        SafeTensorsStreamWriter.Write(Path.Combine(folder, "model-00002-of-00002.safetensors"), blocks);
    }

    public const int DistilledSteps = 4;

    /// <summary>A causal run of <see cref="Steps"/> steps at shift 1, or a distilled run of four steps at shift 5 with re-noise.</summary>
    public static void WriteRun(string folder, bool distilled)
    {
        Directory.CreateDirectory(folder);
        WriteFloats(folder, "shared.bin", Values(2, SharedRows * TextDim));
        WriteFloats(folder, "reference.bin", Values(30, RefRows * VideoIn));
        List<string> blocks = [];
        for (int block = 0; block < Blocks; block++)
        {
            WriteFloats(folder, $"body{block}.bin", Values(40 + block, BodyRows * TextDim));
            WriteFloats(folder, $"video{block}.bin", Values(50 + block, VideoRows * VideoIn));
            WriteFloats(folder, $"audio{block}.bin", Values(60 + block, AudioRows * AudioIn));
            string renoise = "";
            if (distilled)
            {
                WriteFloats(folder, $"rvideo{block}.bin", Values(70 + block, (DistilledSteps - 1) * VideoRows * VideoIn));
                WriteFloats(folder, $"raudio{block}.bin", Values(80 + block, (DistilledSteps - 1) * AudioRows * AudioIn));
                renoise = $$""", "renoiseVideo": "rvideo{{block}}.bin", "renoiseAudio": "raudio{{block}}.bin" """;
            }
            blocks.Add($$"""{ "bodyText": "body{{block}}.bin", "noiseVideo": "video{{block}}.bin", "noiseAudio": "audio{{block}}.bin"{{renoise}} }""");
        }
        File.WriteAllText(Path.Combine(folder, "manifest.json"),
            $$"""{ "steps": {{(distilled ? DistilledSteps : Steps)}}, "shift": {{(distilled ? "5.0" : "1.0")}}, "sharedText": "shared.bin", "referenceVideo": "reference.bin", "blocks": [ {{string.Join(", ", blocks)}} ] }""");
    }

    private static void WriteFloats(string folder, string name, float[] values)
    {
        byte[] bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(Path.Combine(folder, name), bytes);
    }

    private static float[] Values(int seed, int count, float scale = 1f)
    {
        Random random = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = ((float)random.NextDouble() - 0.5f) * scale;
        }
        return values;
    }

    private static float[] Ones(int count) => Enumerable.Repeat(1f, count).ToArray();

    private static SafeTensorsStreamWriter.Entry F32(string name, long[] shape, float[] values) => Entry(name, shape, values, DType.F32);

    private static SafeTensorsStreamWriter.Entry Bf16(string name, long[] shape, float[] values) => Entry(name, shape, values, DType.BF16);

    private static SafeTensorsStreamWriter.Entry Entry(string name, long[] shape, float[] values, DType dtype) =>
        new(name, dtype, new TensorShape(shape), () =>
        {
            Tensor tensor = new(new TensorShape(shape), DType.F32);
            values.AsSpan().CopyTo(tensor.AsSpan<float>());
            if (dtype == DType.F32)
            {
                return (tensor, true);
            }
            using (tensor)
            {
                return (tensor.CastTo(dtype), true);
            }
        });
}
