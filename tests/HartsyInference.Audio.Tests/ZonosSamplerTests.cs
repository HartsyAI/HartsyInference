using System;
using System.IO;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Sampling;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Verifies the min-p nucleus sampler against the reference distribution on real Zonos logits: the
/// kept-token set must match Python's <c>apply_min_p</c>, and the empirical sample histogram must match the
/// analytical min-p distribution (catches multinomial/RNG bias). Gated on <c>ZONOS_GOLDEN</c>.</summary>
public sealed unsafe class ZonosSamplerTests
{
    private readonly ITestOutputHelper _out;
    public ZonosSamplerTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void MinPSampler_MatchesReferenceDistribution()
    {
        string? golden = Environment.GetEnvironmentVariable("ZONOS_GOLDEN");
        if (string.IsNullOrEmpty(golden) || !File.Exists(Path.Combine(golden, "samp_logits_s0c0.bin")))
        { _out.WriteLine("Skipped: dump samp_logits_s0c0.bin / samp_minp_s0c0.bin first."); return; }

        float[] logits = ReadF32(Path.Combine(golden, "samp_logits_s0c0.bin"));
        float[] refDist = ReadF32(Path.Combine(golden, "samp_minp_s0c0.bin"));   // Python min_p renormalized
        int vocab = logits.Length;

        // Analytical min-p distribution from my softmax + min_p, to compare kept set to Python.
        float[] mine = MinPReference(logits, 0.1f);
        int keepMine = 0, keepRef = 0, mismatch = 0;
        for (int i = 0; i < vocab; i++)
        {
            bool m = mine[i] > 0, r = refDist[i] > 0;
            if (m) keepMine++;
            if (r) keepRef++;
            if (m != r) mismatch++;
        }
        _out.WriteLine($"kept: mine={keepMine} ref={keepRef} setMismatch={mismatch}");

        // Empirical histogram from NucleusSampler over many draws (uniform seeds).
        int draws = 200_000;
        int[] hist = new int[vocab];
        uint rng = DeterministicRng.Seed(12345);
        for (int i = 0; i < draws; i++)
        {
            float[] buf = (float[])logits.Clone();
            int tok = NucleusSampler.Draw(buf, vocab, 1f, 0, 0f, ref rng, -1, 0.1f);
            hist[tok]++;
        }
        // Compare empirical freq to Python's min_p distribution for the kept tokens.
        double maxErr = 0; string worst = "";
        for (int i = 0; i < vocab; i++)
        {
            if (refDist[i] <= 0) continue;
            double emp = hist[i] / (double)draws;
            double err = Math.Abs(emp - refDist[i]);
            if (err > maxErr) { maxErr = err; worst = $"tok{i} emp={emp:F4} ref={refDist[i]:F4}"; }
        }
        // Also report total probability my sampler put on tokens the reference EXCLUDES (should be ~0).
        double leak = 0;
        for (int i = 0; i < vocab; i++) if (refDist[i] <= 0) leak += hist[i] / (double)draws;
        _out.WriteLine($"empirical vs ref: maxErr={maxErr:F4} ({worst}); prob-mass on ref-excluded tokens={leak:F4}");

        Assert.True(mismatch == 0, $"min_p kept-set mismatch: {mismatch} tokens");
        Assert.True(leak < 0.01, $"sampler leaks {leak:P1} onto reference-excluded tokens");
        Assert.True(maxErr < 0.02, $"empirical distribution off by {maxErr}");
    }

    private static float[] MinPReference(float[] logits, float minP)
    {
        int n = logits.Length;
        float max = float.NegativeInfinity;
        for (int i = 0; i < n; i++) if (logits[i] > max) max = logits[i];
        float[] p = new float[n]; double sum = 0;
        for (int i = 0; i < n; i++) { p[i] = MathF.Exp(logits[i] - max); sum += p[i]; }
        for (int i = 0; i < n; i++) p[i] = (float)(p[i] / sum);
        float pmax = 0; for (int i = 0; i < n; i++) if (p[i] > pmax) pmax = p[i];
        float thr = minP * pmax; double kept = 0;
        for (int i = 0; i < n; i++) { if (p[i] < thr) p[i] = 0; else kept += p[i]; }
        for (int i = 0; i < n; i++) p[i] = (float)(p[i] / kept);
        return p;
    }

    private static float[] ReadF32(string path)
    {
        byte[] raw = File.ReadAllBytes(path);
        float[] o = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, o, 0, raw.Length);
        return o;
    }
}
