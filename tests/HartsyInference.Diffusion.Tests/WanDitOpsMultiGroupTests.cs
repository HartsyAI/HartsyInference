using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Numerical equivalence of the multi-group (G&gt;1) device paths in <see cref="WanDitOps"/> against the
/// host reference math they replaced (2026-07-11 S2V host-glue port) — S2V/TI2V/Matrix-Game take these branches.</summary>
public unsafe class WanDitOpsMultiGroupTests
{
    private const int Dim = 32;
    private const int FreqDim = 16;

    private static Tensor RandomTensor(TensorShape shape, int seed)
    {
        Tensor t = new Tensor(shape, DType.F32);
        Random rng = new Random(seed);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return t;
    }

    [Fact]
    public void FinalLayer_MultiGroup_MatchesHostReference()
    {
        using CpuBackend backend = new();
        int g = 3, tokensPerGroup = 5, s = g * tokensPerGroup, outVec = 8;
        float eps = 1e-6f;
        using Tensor hidden = RandomTensor(new TensorShape(s, Dim), 1);
        using Tensor temb = RandomTensor(new TensorShape(g, Dim), 2);
        using Tensor scaleShift = RandomTensor(new TensorShape(2, Dim), 3);
        using Tensor projW = RandomTensor(new TensorShape(outVec, Dim), 4);
        using Tensor projB = RandomTensor(new TensorShape(outVec), 5);

        // Device path (g·tokensPerGroup == s).
        using Tensor got = WanDitOps.FinalLayer(backend, hidden, temb, scaleShift, projW, projB, s, Dim, eps, tokensPerGroup);

        // Host reference: per-token no-affine LN, per-group (1+scale)·x+shift from scale_shift_table+temb, proj.
        float* hp = (float*)hidden.DataPointer;
        float* ss = (float*)scaleShift.DataPointer;
        float* em = (float*)temb.DataPointer;
        using Tensor modded = new Tensor(new TensorShape(s, Dim), DType.F32);
        float* mp = (float*)modded.DataPointer;
        for (int i = 0; i < s; i++)
        {
            double mean = 0;
            for (int d = 0; d < Dim; d++) mean += hp[i * Dim + d];
            mean /= Dim;
            double var = 0;
            for (int d = 0; d < Dim; d++) { double diff = hp[i * Dim + d] - mean; var += diff * diff; }
            var /= Dim;
            float invStd = 1f / MathF.Sqrt((float)var + eps);
            int gi = i / tokensPerGroup;
            for (int d = 0; d < Dim; d++)
            {
                float normed = (hp[i * Dim + d] - (float)mean) * invStd;
                float scale = ss[Dim + d] + em[gi * Dim + d];
                float shift = ss[d] + em[gi * Dim + d];
                mp[i * Dim + d] = normed * (1f + scale) + shift;
            }
        }
        using Tensor want = new Tensor(new TensorShape(s, outVec), DType.F32);
        backend.Linear(want, modded, projW, projB);

        float* gp = (float*)got.DataPointer;
        float* wp = (float*)want.DataPointer;
        for (long i = 0; i < want.ElementCount; i++)
            Assert.True(MathF.Abs(gp[i] - wp[i]) < 1e-4f, $"FinalLayer mismatch at {i}: {gp[i]} vs {wp[i]}");
    }

    [Fact]
    public void ConditionTimeGroups_MultiGroup_MatchesPerGroupReference()
    {
        using CpuBackend backend = new();
        float[] timesteps = [999f, 999f, 0f];
        int g = timesteps.Length;
        using Tensor emb1W = RandomTensor(new TensorShape(Dim, FreqDim), 10);
        using Tensor emb1B = RandomTensor(new TensorShape(Dim), 11);
        using Tensor emb2W = RandomTensor(new TensorShape(Dim, Dim), 12);
        using Tensor emb2B = RandomTensor(new TensorShape(Dim), 13);
        using Tensor projW = RandomTensor(new TensorShape(6 * Dim, Dim), 14);
        using Tensor projB = RandomTensor(new TensorShape(6 * Dim), 15);

        (Tensor temb, Tensor proj) = WanDitOps.ConditionTimeGroups(backend, timesteps, FreqDim, Dim,
            emb1W, emb1B, emb2W, emb2B, projW, projB);
        using Tensor _t = temb; using Tensor _p = proj;

        // Reference: the G==1 path run per group must produce each group's rows bit-identically (the device
        // Concat gather must not change values vs the old per-group host copy).
        for (int gi = 0; gi < g; gi++)
        {
            (Tensor temb1, Tensor proj1) = WanDitOps.ConditionTimeGroups(backend, [timesteps[gi]], FreqDim, Dim,
                emb1W, emb1B, emb2W, emb2B, projW, projB);
            using Tensor _t1 = temb1; using Tensor _p1 = proj1;
            float* tg = (float*)temb.DataPointer + (long)gi * Dim;
            float* t1 = (float*)temb1.DataPointer;
            for (int d = 0; d < Dim; d++)
                Assert.True(tg[d] == t1[d], $"temb group {gi} mismatch at {d}: {tg[d]} vs {t1[d]}");
            float* pg = (float*)proj.DataPointer + (long)gi * 6 * Dim;
            float* p1 = (float*)proj1.DataPointer;
            for (int d = 0; d < 6 * Dim; d++)
                Assert.True(pg[d] == p1[d], $"proj group {gi} mismatch at {d}: {pg[d]} vs {p1[d]}");
        }
    }
}
