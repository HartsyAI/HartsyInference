using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight Boogu-Image Edit end-to-end repro/verify harness (mirrors the SwarmUI extension's
/// <c>BooguImageLoader.GenerateEdit</c> flow). It reproduced the 2026-07-16 "instruction ignored" bug (1024-token
/// VLM conditioning from a native-res init image) and verified the fix (reference 384·384 VLM budget → yellow-pear
/// edit; see <c>docs/Research/BOOGU_IMAGE.md</c> §5). Env-gated: needs <c>HARTSY_BOOGU_EDIT_DIT</c> /
/// <c>HARTSY_BOOGU_QWEN3VL</c> / <c>HARTSY_BOOGU_VAE</c> / <c>HARTSY_BOOGU_IMG768</c> (square HWC u8 raw, any side) /
/// <c>HARTSY_BOOGU_OUT_DIR</c>. Knobs: <c>HARTSY_BOOGU_VLM_MAXPIX</c> (0 = processor default budget; 147456 =
/// reference), <c>HARTSY_BOOGU_NEG_IMG</c> (1 = negative encode sees the image — the buggy loader behavior; 0 =
/// reference), <c>HARTSY_BOOGU_STAGE</c> (0 = no TE⇄DiT staging), <c>HARTSY_BOOGU_STEPS</c> / <c>HARTSY_BOOGU_TG</c> /
/// <c>HARTSY_BOOGU_GEN</c> / <c>HARTSY_BOOGU_SEED</c>. Runs on CUDA device 0.</summary>
public sealed class BooguImageEditEndToEndTests
{
    private readonly ITestOutputHelper _output;
    public BooguImageEditEndToEndTests(ITestOutputHelper output) => _output = output;

    private const int VisionStartTokenId = 151652;
    private const int VisionEndTokenId = 151653;
    private const int ImagePadTokenId = 151655;

    [Fact]
    [Trait("Category", "Slow")]
    public unsafe void EditFollowsInstruction()
    {
        string? ditPath = Environment.GetEnvironmentVariable("HARTSY_BOOGU_EDIT_DIT");
        string? tePath = Environment.GetEnvironmentVariable("HARTSY_BOOGU_QWEN3VL");
        string? vaePath = Environment.GetEnvironmentVariable("HARTSY_BOOGU_VAE");
        string? imgPath = Environment.GetEnvironmentVariable("HARTSY_BOOGU_IMG768");
        string? outDir = Environment.GetEnvironmentVariable("HARTSY_BOOGU_OUT_DIR");
        if (ditPath is null || tePath is null || vaePath is null || imgPath is null || outDir is null)
            return;
        Directory.CreateDirectory(outDir);

        int vlmMaxPix = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_BOOGU_VLM_MAXPIX"), out int mp) ? mp : 0;
        bool negSeesImage = Environment.GetEnvironmentVariable("HARTSY_BOOGU_NEG_IMG") != "0";
        int steps = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_BOOGU_STEPS"), out int st) ? st : 15;
        float tg = float.TryParse(Environment.GetEnvironmentVariable("HARTSY_BOOGU_TG"), out float tgv) ? tgv : 4.0f;
        bool stage = Environment.GetEnvironmentVariable("HARTSY_BOOGU_STAGE") != "0";
        int gen = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_BOOGU_GEN"), out int g) ? g : 512;
        const string prompt = "replace the apple with a yellow pear";
        const string negative = "";

        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        using CudaBackend backend = new(0, ptxDir);
        Stopwatch sw = Stopwatch.StartNew();

        // ── components (same key transforms as BooguImageLoader) ──
        (Dictionary<string, Tensor> ditW, SafeTensorsLoader ditL) = LoadComponent(ditPath, StripTransformerPrefix, fp8: true);
        (Dictionary<string, Tensor> teW, SafeTensorsLoader teL) = LoadComponent(tePath, RemapQwenLanguageKey, fp8: true);
        (Dictionary<string, Tensor> visW, SafeTensorsLoader visL) = LoadComponent(tePath, RemapQwenVisionKey, fp8: true);
        (Dictionary<string, Tensor> vaeW, SafeTensorsLoader vaeL) = LoadComponent(vaePath,
            k => CheckpointConvertUtils.ConvertVaeKey(k.StartsWith("vae.", StringComparison.Ordinal) ? k[4..] : k), fp8: false);
        _output.WriteLine($"weights loaded in {sw.ElapsedMilliseconds}ms (dit={ditW.Count}, te={teW.Count}, vis={visW.Count}, vae={vaeW.Count})");

        BooguImageConfig config = BooguImageConfig.V01;
        BooguImageTransformer transformer = new(config);
        transformer.LoadWeights(ditW);
        LlamaStyleEncoder textEncoder = new(LlamaStyleEncoderConfig.Qwen3_VL_8B);
        textEncoder.LoadWeights(teW);
        VaeDecoder vaeDecoder = new(VaeConfig.Flux);
        vaeDecoder.LoadWeights(vaeW);
        VaeEncoder vaeEncoder = new(VaeConfig.Flux);
        vaeEncoder.LoadWeights(vaeW);

        Qwen3VlVisionConfig visionConfig = Qwen3VlVisionConfig.Qwen3Vl8B;
        Qwen3VlVisionEncoder visionEncoder = new(visionConfig);
        visionEncoder.LoadWeights(visW);
        Qwen3VlImageProcessor imageProcessor = vlmMaxPix > 0
            ? new Qwen3VlImageProcessor(visionConfig, maxPixels: vlmMaxPix)
            : new Qwen3VlImageProcessor(visionConfig);
        Qwen3VlMultimodalEncoder multimodal = new(textEncoder, visionEncoder, imageProcessor, visionConfig,
            imageTokenId: ImagePadTokenId, textHeadDim: 128, ropeTheta: 5_000_000.0, mropeSection: [24, 20, 20]);
        Qwen3Tokenizer tokenizer = new(maxLength: 4096);

        // ── reference image (square HWC u8 raw; side inferred from length) in both forms ──
        byte[] rgbSrc = File.ReadAllBytes(imgPath);
        int side = (int)Math.Sqrt(rgbSrc.Length / 3.0);
        Assert.Equal(side * side * 3, rgbSrc.Length);
        int seed = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_BOOGU_SEED"), out int sd) ? sd : 42;
        _output.WriteLine($"init image {side}x{side}, seed={seed}");
        using Tensor visionRgb = HwcRgbToChw01(rgbSrc, side, side);
        byte[] rgbGen = ResizeHwc(rgbSrc, side, side, gen, gen);
        using Tensor refLatentInput = ImagePostProcessor.RgbBytesToTensor(rgbGen, gen, gen);

        (int rh, int rw) = imageProcessor.SmartResize(side, side);
        int mergeFactor = visionConfig.PatchSize * visionConfig.SpatialMergeSize;
        int numMerged = (rh / mergeFactor) * (rw / mergeFactor);
        _output.WriteLine($"vlm image: {rh}x{rw} → {numMerged} merged tokens; negSeesImage={negSeesImage}, tg={tg}, steps={steps}");

        int[] condTokens = BuildTokens(tokenizer, prompt, numMerged);
        int[] dropTokens = negSeesImage ? BuildTokens(tokenizer, negative, numMerged) : BuildTokens(tokenizer, negative, 0);
        _output.WriteLine($"cond tokens={condTokens.Length}, dropText tokens={dropTokens.Length}");

        sw.Restart();
        Tensor cond = multimodal.Encode(backend, condTokens, [visionRgb]);
        Tensor dropText = negSeesImage
            ? multimodal.Encode(backend, dropTokens, [visionRgb])
            : multimodal.Encode(backend, dropTokens, []);
        backend.Sync();
        _output.WriteLine($"encodes in {sw.ElapsedMilliseconds}ms; cond {cond.Shape}, dropText {dropText.Shape}");
        EmbedStats("cond", cond);
        EmbedStats("dropText", dropText);

        // TE ⇄ DiT staging (the loader's T2I pattern): free the TE weights, host-materialize the embeddings.
        // HARTSY_BOOGU_STAGE=0 skips this, matching the current loader edit path (loader-exact repro).
        if (stage)
        {
            _ = cond.DataPointer;
            _ = dropText.DataPointer;
            backend.Sync();
            backend.FreeWeights(textEncoder.EnumerateWeights());
            backend.FreeWeights(visionEncoder.EnumerateWeights());
            backend.FreeActivations();
        }

        BooguImagePipeline pipeline = new(backend, transformer, vaeDecoder, vaeEncoder, config);
        TextToImageRequest request = new()
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = gen,
            Height = gen,
            Steps = steps,
            CfgScale = tg,
            Seed = seed,
        };

        sw.Restart();
        (byte[] outRgb, int ow, int oh, _) = pipeline.EditFromEmbeddings(cond, dropText, null, [refLatentInput],
            request, tg, 1.0f, p => _output.WriteLine($"  step {p.Step}/{p.TotalSteps} {p.ElapsedMs:F0}ms"));
        _output.WriteLine($"edit {ow}x{oh} in {sw.ElapsedMilliseconds}ms");

        string tag = $"maxpix{vlmMaxPix}_neg{(negSeesImage ? 1 : 0)}_tg{tg}_s{steps}_g{gen}_st{(stage ? 1 : 0)}_i{side}_sd{seed}";
        File.WriteAllBytes(Path.Combine(outDir, $"edit_{tag}.bin"), outRgb);
        File.WriteAllText(Path.Combine(outDir, $"edit_{tag}.txt"), $"{ow} {oh}");
        _output.WriteLine($"wrote {outDir}/edit_{tag}.bin");

        cond.Dispose();
        dropText.Dispose();
        pipeline.Dispose();
        ditL.Dispose(); teL.Dispose(); visL.Dispose(); vaeL.Dispose();
    }

    private unsafe void EmbedStats(string name, Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        int seq = (int)t.Shape[1];
        int dim = (int)t.Shape[2];
        double sumSq = 0;
        int nonFinite = 0;
        for (long i = 0; i < n; i++)
        {
            float v = p[i];
            if (!float.IsFinite(v)) nonFinite++;
            else sumSq += (double)v * v;
        }
        // per-position norms at start / middle / end
        double NormAt(int s)
        {
            double acc = 0;
            for (int d = 0; d < dim; d++) { double v = p[(long)s * dim + d]; acc += v * v; }
            return Math.Sqrt(acc);
        }
        _output.WriteLine($"{name}: rms={Math.Sqrt(sumSq / n):F3} nonFinite={nonFinite} " +
            $"norm[0]={NormAt(0):F1} norm[mid]={NormAt(seq / 2):F1} norm[last]={NormAt(seq - 1):F1}");
    }

    private static int[] BuildTokens(Qwen3Tokenizer tok, string instruction, int numImagePad)
    {
        const string system = "Describe the key features of the input image (color, shape, size, texture, objects, background), " +
            "then explain how the user's text instruction should alter or modify the image. Generate a new image that meets " +
            "the user's requirements while maintaining consistency with the original input where appropriate.";
        List<int> ids = new(256);
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tok.EncodeRaw("system\n" + system));
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tok.EncodeRaw("\n"));
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tok.EncodeRaw("user\n"));
        if (numImagePad > 0)
        {
            ids.Add(VisionStartTokenId);
            for (int i = 0; i < numImagePad; i++) ids.Add(ImagePadTokenId);
            ids.Add(VisionEndTokenId);
        }
        ids.AddRange(tok.EncodeRaw(instruction ?? ""));
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tok.EncodeRaw("\n"));
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tok.EncodeRaw("assistant\n"));
        return [.. ids];
    }

    private static unsafe Tensor HwcRgbToChw01(byte[] rgb, int width, int height)
    {
        Tensor t = new(new TensorShape(3, height, width), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int src = (y * width + x) * 3;
                for (int c = 0; c < 3; c++)
                    p[(long)c * height * width + (long)y * width + x] = rgb[src + c] / 255.0f;
            }
        return t;
    }

    /// <summary>Nearest-neighbor HWC u8 resize (test-only; the extension uses its own resizer).</summary>
    private static byte[] ResizeHwc(byte[] src, int sw, int sh, int dw, int dh)
    {
        byte[] dst = new byte[dw * dh * 3];
        for (int y = 0; y < dh; y++)
        {
            int sy = (int)((y + 0.5) * sh / dh);
            for (int x = 0; x < dw; x++)
            {
                int sx = (int)((x + 0.5) * sw / dw);
                int s = (sy * sw + sx) * 3, d = (y * dw + x) * 3;
                dst[d] = src[s]; dst[d + 1] = src[s + 1]; dst[d + 2] = src[s + 2];
            }
        }
        return dst;
    }

    private static (Dictionary<string, Tensor>, SafeTensorsLoader) LoadComponent(string path, Func<string, string?> map, bool fp8)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> result = new();
        foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
        {
            if (kvp.Key.EndsWith(".scaled_fp8", StringComparison.Ordinal) || kvp.Key == "scaled_fp8") continue;
            string? mapped = map(kvp.Key);
            if (mapped is not null) result[mapped] = kvp.Value;
        }
        return (fp8 ? CheckpointConvertUtils.ApplyFp8ScaledDequant(result) : result, loader);
    }

    private static string StripTransformerPrefix(string key)
    {
        if (key.StartsWith("model.diffusion_model.", StringComparison.Ordinal)) return key["model.diffusion_model.".Length..];
        if (key.StartsWith("diffusion_model.", StringComparison.Ordinal)) return key["diffusion_model.".Length..];
        if (key.StartsWith("transformer.", StringComparison.Ordinal)) return key["transformer.".Length..];
        return key;
    }

    private static string? RemapQwenLanguageKey(string key)
    {
        if (key.Contains(".visual.") || key.StartsWith("visual.", StringComparison.Ordinal)) return null;
        if (key.Contains("lm_head")) return null;
        int lm = key.LastIndexOf("language_model.", StringComparison.Ordinal);
        string suffix = lm >= 0 ? key[(lm + "language_model.".Length)..] : key;
        if (suffix.StartsWith("model.", StringComparison.Ordinal)) suffix = suffix["model.".Length..];
        if (suffix.StartsWith("layers.", StringComparison.Ordinal)
            || suffix.StartsWith("embed_tokens.", StringComparison.Ordinal)
            || suffix == "norm.weight")
            return "model." + suffix;
        return null;
    }

    private static string? RemapQwenVisionKey(string key)
    {
        int v = key.LastIndexOf(".visual.", StringComparison.Ordinal);
        if (v >= 0) return key[(v + ".visual.".Length)..];
        if (key.StartsWith("visual.", StringComparison.Ordinal)) return key["visual.".Length..];
        return null;
    }
}
