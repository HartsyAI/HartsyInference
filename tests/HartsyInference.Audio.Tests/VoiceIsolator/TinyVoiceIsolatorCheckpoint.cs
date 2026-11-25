using System.Text.Json;
using HartsyInference.Audio.Models.VoiceIsolator;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Audio.Tests.VoiceIsolator;

/// <summary>A small random-weight isolator checkpoint written as safetensors, with the tensor names and shapes of the released files.</summary>
internal static unsafe class TinyVoiceIsolatorCheckpoint
{
    public const int Width = 16;
    public const int Heads = 2;
    public const int NFft = 64;
    public const int Hop = 16;
    public const int State = 4;

    public static string ConfigJson(bool causal, int conv = 4) => JsonSerializer.Serialize(new Dictionary<string, object>
    {
        ["width"] = Width,
        ["block_types"] = new[] { "transformer", "mamba", "transformer" },
        ["num_heads"] = Heads,
        ["n_fft"] = NFft,
        ["hop_length"] = Hop,
        ["sample_rate"] = 1600,
        ["band_plan"] = new[] { new[] { 100, 400 }, new[] { 200, 800 } },
        ["correction_bound"] = 0.5,
        ["chunk_seconds"] = 1.0,
        ["overlap_seconds"] = 0.25,
        ["exclusive_self_attention"] = true,
        ["exclusive_axes"] = "time",
        ["mamba_state"] = State,
        ["mamba_conv"] = conv,
        ["mamba_expand"] = 2,
        ["causal"] = causal,
        ["lookahead_frames"] = causal ? 2 : 0,
        ["lookahead_frames_choices"] = causal ? new[] { 0, 2, 4 } : Array.Empty<int>(),
        ["attention_window_frames"] = causal ? 6 : 0,
    });

    public static string Write(string directory, bool causal, int seed, int metadataConv = 4)
    {
        VoiceIsolatorConfig config = VoiceIsolatorConfig.FromJson(ConfigJson(causal));
        (int Start, int End)[] edges = VoiceIsolatorBands.Edges(config);
        Random rng = new(seed);
        Dictionary<string, Tensor> t = new();
        int w = config.Width;
        int inner = config.MambaInner;
        int dtRank = config.MambaDtRank;
        t["band_embedding"] = Random2(rng, edges.Length, w);
        if (causal)
        {
            t["lookahead_embedding.weight"] = Random2(rng, 3, w);
        }
        for (int b = 0; b < edges.Length; b++)
        {
            int count = 2 * (edges[b].End - edges[b].Start);
            Norm(t, $"encoders.{b}.0", count, rng);
            Linear(t, $"encoders.{b}.1", w, count, rng);
            Norm(t, $"denoise_decoder.heads.{b}.0", w, rng);
            Linear(t, $"denoise_decoder.heads.{b}.1", w, w, rng);
            Linear(t, $"denoise_decoder.heads.{b}.3", 4 * (edges[b].End - edges[b].Start), w, rng);
        }
        Linear(t, "denoise_decoder.condition", 2 * w, 2, rng);
        for (int i = 0; i < config.BlockTypes.Count; i++)
        {
            string p = $"blocks.{i}";
            if (config.BlockTypes[i] == "transformer")
            {
                Norm(t, $"{p}.time_transformer.norm1", w, rng);
                Norm(t, $"{p}.time_transformer.norm2", w, rng);
                Linear(t, $"{p}.time_transformer.attn.qkv", 3 * w, w, rng);
                Linear(t, $"{p}.time_transformer.attn.proj", w, w, rng);
                Linear(t, $"{p}.time_transformer.ffn.0", 4 * w, w, rng);
                Linear(t, $"{p}.time_transformer.ffn.3", w, 4 * w, rng);
                Norm(t, $"{p}.band_transformer.norm1", w, rng);
                Norm(t, $"{p}.band_transformer.norm2", w, rng);
                t[$"{p}.band_transformer.self_attn.in_proj_weight"] = Random2(rng, 3 * w, w);
                t[$"{p}.band_transformer.self_attn.in_proj_bias"] = Random1(rng, 3 * w);
                Linear(t, $"{p}.band_transformer.self_attn.out_proj", w, w, rng);
                Linear(t, $"{p}.band_transformer.linear1", 4 * w, w, rng);
                Linear(t, $"{p}.band_transformer.linear2", w, 4 * w, rng);
            }
            else
            {
                MambaPair(t, $"{p}.time_mamba", causal, config, rng);
                MambaPair(t, $"{p}.freq_mamba", false, config, rng);
            }
        }
        string path = Path.Combine(directory, causal ? "tiny_causal.safetensors" : "tiny_offline.safetensors");
        SafeTensorsWriter.Save(path, t, new Dictionary<string, string> { ["config"] = ConfigJson(causal, metadataConv) });
        foreach (Tensor tensor in t.Values)
        {
            tensor.Dispose();
        }
        _ = (inner, dtRank);
        return path;
    }

    private static void MambaPair(Dictionary<string, Tensor> t, string p, bool causal, VoiceIsolatorConfig c, Random rng)
    {
        MambaLayer(t, $"{p}.forward_blocks", c, rng);
        if (!causal)
        {
            MambaLayer(t, $"{p}.backward_blocks", c, rng);
        }
        Linear(t, $"{p}.output_proj", c.Width, causal ? c.Width : 2 * c.Width, rng);
        Norm(t, $"{p}.norm", c.Width, rng);
    }

    private static void MambaLayer(Dictionary<string, Tensor> t, string p, VoiceIsolatorConfig c, Random rng)
    {
        int inner = c.MambaInner;
        t[$"{p}.in_proj.weight"] = Random2(rng, 2 * inner, c.Width);
        t[$"{p}.conv1d.weight"] = Random3(rng, inner, 1, c.MambaConv);
        t[$"{p}.conv1d.bias"] = Random1(rng, inner);
        t[$"{p}.x_proj.weight"] = Random2(rng, c.MambaDtRank + 2 * c.MambaState, inner);
        t[$"{p}.dt_proj.weight"] = Random2(rng, inner, c.MambaDtRank);
        t[$"{p}.dt_proj.bias"] = Random1(rng, inner);
        Tensor aLog = new(new TensorShape(inner, c.MambaState), DType.F32);
        float* a = (float*)aLog.DataPointer;
        for (int i = 0; i < inner; i++)
        {
            for (int n = 0; n < c.MambaState; n++)
            {
                a[i * c.MambaState + n] = MathF.Log(n + 1);
            }
        }
        t[$"{p}.A_log"] = aLog;
        Tensor d = Random1(rng, inner);
        t[$"{p}.D"] = d;
        t[$"{p}.out_proj.weight"] = Random2(rng, c.Width, inner);
    }

    private static void Norm(Dictionary<string, Tensor> t, string p, int width, Random rng)
    {
        Tensor weight = Random1(rng, width);
        float* wp = (float*)weight.DataPointer;
        for (int i = 0; i < width; i++)
        {
            wp[i] += 1f;
        }
        t[$"{p}.weight"] = weight;
        t[$"{p}.bias"] = Random1(rng, width);
    }

    private static void Linear(Dictionary<string, Tensor> t, string p, int outWidth, int inWidth, Random rng)
    {
        t[$"{p}.weight"] = Random2(rng, outWidth, inWidth);
        t[$"{p}.bias"] = Random1(rng, outWidth);
    }

    private static Tensor Random1(Random rng, int a) => Fill(new Tensor(new TensorShape(a), DType.F32), rng);

    private static Tensor Random2(Random rng, int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32), rng);

    private static Tensor Random3(Random rng, int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32), rng);

    private static Tensor Fill(Tensor tensor, Random rng)
    {
        float* p = (float*)tensor.DataPointer;
        for (long i = 0; i < tensor.ElementCount; i++)
        {
            p[i] = ((float)rng.NextDouble() - 0.5f) * 0.4f;
        }
        return tensor;
    }
}
