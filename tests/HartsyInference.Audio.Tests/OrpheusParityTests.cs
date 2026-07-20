using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.Orpheus;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight parity for the Orpheus-3B LM backbone (Llama-3.2-3B → <see cref="Qwen2Model"/>) on CUDA,
/// vs the upstream <c>LlamaForCausalLM</c> (unsloth/orpheus-3b-0.1-ft mirror). Teacher-forced logits over a fixed
/// token sequence; checks per-position greedy argmax (exact) and logit correlation. The codec half (SNAC) is
/// already verified separately. Gated on CUDA + <c>ORPHEUS_DIR</c> (safetensors dir) + <c>ORPHEUS_REF</c>
/// (dump_orpheus_reference.py). bf16 weights stay GPU-resident (~6 GB VRAM); never f32 on the host.</summary>
public sealed unsafe class OrpheusParityTests
{
    private readonly ITestOutputHelper _out;
    public OrpheusParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void OrpheusBackbone_Logits_MatchReference()
    {
        string? dir = Environment.GetEnvironmentVariable("ORPHEUS_DIR");
        string? refP = Environment.GetEnvironmentVariable("ORPHEUS_REF");
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir) || string.IsNullOrEmpty(refP) || !File.Exists(refP))
            return;
        if (!CudaContext.IsAvailable()) { _out.WriteLine("SKIPPED: CUDA unavailable"); return; }

        // Merge all shards into one weight dict.
        Dictionary<string, Tensor> w = new();
        foreach (string f in Directory.GetFiles(dir, "*.safetensors"))
        {
            SafeTensorsLoader l = new(); l.Load(f);
            foreach (KeyValuePair<string, Tensor> kv in l.GetAllTensors()) w[kv.Key] = kv.Value;
        }
        _out.WriteLine($"loaded {w.Count} tensors");

        SafeTensorsLoader rl = new(); rl.Load(refP);
        IReadOnlyDictionary<string, Tensor> r = rl.GetAllTensors();
        Tensor idsT = r["input_ids"];          // [1, T] int32
        Tensor logitsRef = r["logits"];        // [1, T, vocab]
        Tensor argmaxRef = r["argmax"];        // [T]
        int t = (int)idsT.Shape[1];
        int vocab = (int)logitsRef.Shape[2];
        int[] ids = new int[t];
        int* ip = (int*)idsT.DataPointer;
        for (int i = 0; i < t; i++) ids[i] = ip[i];

        Qwen2Config cfg = OrpheusConfig.Orpheus3B.Llm;
        Qwen2Model model = new(cfg);
        model.LoadWeights(w, "model");

        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        using CudaBackend backend = new(0, ptxDir);
        using StreamingKvCache cache = new(cfg.NumHiddenLayers, batch: 1, cfg.NumKeyValueHeads, t + 4, cfg.HeadDim);
        Tensor hidden = model.Forward(backend, ids, batch: 1, posStart: 0, cache);
        Tensor logits = model.ProjectLogits(backend, hidden, batch: 1, t);
        hidden.Dispose();

        // Per-position greedy argmax (functional check) + correlation (bf16 ref → ~1e-2 noise).
        float* lp = (float*)logits.DataPointer;
        int* amRef = (int*)argmaxRef.DataPointer;
        int argmaxHits = 0;
        for (int pos = 0; pos < t; pos++)
        {
            long off = (long)pos * vocab;
            int am = 0; float best = lp[off];
            for (int k = 1; k < vocab; k++) { float v = lp[off + k]; if (v > best) { best = v; am = k; } }
            if (am == amRef[pos]) argmaxHits++;
        }
        (double corr, double mx) = Compare(logits, logitsRef);
        _out.WriteLine($"Orpheus logits corr={corr:F6} maxAbs={mx:E3}  argmax {argmaxHits}/{t}");
        logits.Dispose();
        model.Dispose();

        Assert.Equal(t, argmaxHits);
        Assert.True(corr > 0.99, $"Orpheus logits corr too low ({corr:F6}).");
    }

    private static (double corr, double mx) Compare(Tensor mine, Tensor refT)
    {
        long n = Math.Min(mine.ElementCount, refT.ElementCount);
        float* a = (float*)mine.DataPointer; float* b = (float*)refT.DataPointer;
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, mx = 0;
        for (long i = 0; i < n; i++)
        {
            double x = a[i], y = b[i];
            sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
            mx = Math.Max(mx, Math.Abs(x - y));
        }
        double cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
        return (cov / (Math.Sqrt(va * vb) + 1e-12), mx);
    }
}
