using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.HeartMula;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Numerical parity for the HeartMuLa-oss-3B LM (CSM-shaped, torchtune <c>llama3_2</c>) against the
/// upstream <c>heartlib</c>. The Python oracle (<c>/tmp/heartmula_ref/oracle.py</c>) reconstructs the model from
/// source, loads the real f32 checkpoint as bf16, runs <c>generate_frame</c> on a fixed lyrics-token grid +
/// teacher-forced codebook values, and dumps the codebook-0 logits + the depth decoder's cb1..7 logits to
/// <c>heartmula_lm_ref.safetensors</c>. This loads the same checkpoint (cast bf16 on CUDA with
/// <c>HighPrecisionGemm</c>), runs <see cref="HeartMulaPipeline.DebugFrameLogits"/> with identical inputs, and
/// checks each argmax matches the reference (allowing a near-tie flip, since the reference is itself bf16) and
/// that the logits correlate &gt; 0.99.
///
/// <para>Gated on <c>HEARTMULA_LM_WEIGHTS_DIR</c> (a dir with the 4 sharded safetensors) +
/// <c>HEARTMULA_LM_REF</c> (the dumped reference) + <c>HEARTMULA_PTX</c> (CUDA — the 3B bf16 needs the GPU).</para></summary>
public sealed unsafe class HeartMulaLmParityTests
{
    private static readonly int[] TextIds = [128000, 100, 200, 300, 400, 128001];
    private static readonly int[] ForcedCodes = [11, 22, 33, 44, 55, 66, 77, 88];

    private readonly ITestOutputHelper _out;
    public HeartMulaLmParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void Lm_MatchesHeartlibReference()
    {
        string? dir = Environment.GetEnvironmentVariable("HEARTMULA_LM_WEIGHTS_DIR");
        string? refP = Environment.GetEnvironmentVariable("HEARTMULA_LM_REF");
        string? ptx = Environment.GetEnvironmentVariable("HEARTMULA_PTX");
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir) || string.IsNullOrEmpty(refP) || !File.Exists(refP) || string.IsNullOrEmpty(ptx))
            return;

        // Load the sharded f32 checkpoint, cast each tensor to bf16 so the 3B fits the 3060 (HiP upcasts per-GEMM).
        List<SafeTensorsLoader> loaders = new();
        Dictionary<string, Tensor> bf = new();
        foreach (string f in Directory.GetFiles(dir, "*.safetensors"))
        {
            SafeTensorsLoader l = new(); l.Load(f); loaders.Add(l);
            foreach ((string k, Tensor t) in l.GetAllTensors())
                bf[k] = t.DType == DType.F32 ? t.CastTo(DType.BF16) : t;
        }

        using SafeTensorsLoader refLoader = new(); refLoader.Load(refP);
        IReadOnlyDictionary<string, Tensor> rd = refLoader.GetAllTensors();
        Tensor c0Ref = rd["c0_logits"];     // [1, vocab]
        Tensor decRef = rd["dec_logits"];   // [7, vocab]
        int vocab = (int)c0Ref.Shape[c0Ref.Shape.Rank - 1];

        using CudaBackendOrCpu backend = new(ptx);
        using HeartMulaPipeline pipe = new(HeartMulaConfig.Oss3B);
        pipe.LoadWeights(bf);

        (float[] c0, float[][] dec) = pipe.DebugFrameLogits(backend.Backend, TextIds, ForcedCodes);

        float* c0r = (float*)c0Ref.DataPointer;
        (double corr, int refMax, int csMax, double gap) = Compare(c0, c0r, vocab);
        _out.WriteLine($"c0: corr={corr:F6} ref-argmax={refMax} cs-argmax={csMax} (top-2 gap {gap:F3})");
        Assert.True(corr > 0.99, $"c0 logits diverge (corr={corr:F4}).");
        Assert.True(refMax == csMax || gap < 0.05, $"c0 argmax {csMax} != ref {refMax} with a non-tie gap {gap:F3}.");

        float* decr = (float*)decRef.DataPointer;
        for (int i = 0; i < dec.Length; i++)
        {
            (double dc, int rm, int cm, double g) = Compare(dec[i], decr + (long)i * vocab, vocab);
            _out.WriteLine($"dec[{i}] cb{i + 1}: corr={dc:F6} ref-argmax={rm} cs-argmax={cm} (top-2 gap {g:F3})");
            Assert.True(dc > 0.99, $"dec[{i}] logits diverge (corr={dc:F4}).");
            Assert.True(rm == cm || g < 0.05, $"dec[{i}] argmax {cm} != ref {rm} with a non-tie gap {g:F3}.");
        }

        foreach (SafeTensorsLoader l in loaders) l.Dispose();
    }

    /// <summary>Returns (Pearson correlation, reference argmax, candidate argmax, reference top-2 gap). A small
    /// reference gap means the reference argmax is itself a bf16 numerical tie, so a flipped candidate argmax is
    /// reference-side noise, not a C# bug. <paramref name="cs"/> is the C# logits, <paramref name="refp"/> the
    /// reference; the gap is computed on the **reference** (the side whose top-1 we are matching against).</summary>
    private static (double Corr, int RefMax, int CsMax, double RefGap) Compare(float[] cs, float* refp, int n)
    {
        double mc = 0, mr = 0;
        for (int i = 0; i < n; i++) { mc += cs[i]; mr += refp[i]; }
        mc /= n; mr /= n;
        double cov = 0, vc = 0, vr = 0;
        int csMax = 0; double csBest = double.NegativeInfinity;
        int refMax = 0; double r1 = double.NegativeInfinity, r2 = double.NegativeInfinity;
        for (int i = 0; i < n; i++)
        {
            double dc = cs[i] - mc, dr = refp[i] - mr;
            cov += dc * dr; vc += dc * dc; vr += dr * dr;
            if (cs[i] > csBest) { csBest = cs[i]; csMax = i; }
            if (refp[i] > r1) { r2 = r1; r1 = refp[i]; refMax = i; } else if (refp[i] > r2) r2 = refp[i];
        }
        double corr = cov / (Math.Sqrt(vc * vr) + 1e-12);
        return (corr, refMax, csMax, r1 - r2);
    }

    /// <summary>Tiny helper so the test body reads cleanly; constructs the CUDA backend with HiP enabled.</summary>
    private sealed class CudaBackendOrCpu : IDisposable
    {
        private readonly HartsyInference.Cuda.CudaBackend _cuda;
        public IBackend Backend => _cuda;
        public CudaBackendOrCpu(string ptx) { _cuda = new HartsyInference.Cuda.CudaBackend(0, ptx); _cuda.HighPrecisionGemm = true; }
        public void Dispose() => _cuda.Dispose();
    }
}
