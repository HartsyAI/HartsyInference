using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Dia;
using HartsyInference.Audio.Pipelines;
using Xunit;

namespace HartsyInference.Audio.Tests.Sampling;

/// <summary>Proves <see cref="DiaPipeline.SampleDiaChannel(float[], float[], int, DiaConfig, ref uint, int, float, float)"/>
/// (called 9x per frame — once per DAC channel) still returns the EXACT token the pre-fix algorithm did.
/// Unlike the other fixes in this PR, this call site's OWN top-K WINDOW selection (not just the later
/// <c>NucleusSampler.Draw</c> fallback it feeds) used a bespoke full delegate sort; <see cref="ReferenceSample"/>
/// keeps that pre-fix selection verbatim as the oracle.</summary>
public sealed class DiaSampleChannelIdentityTests
{
    private static readonly DiaConfig Cfg = DiaConfig.Dia1_6B;

    /// <summary>Verbatim pre-fix <c>DiaPipeline.SampleDiaChannel</c> window selection, kept ONLY as the test
    /// oracle. The tail (EOS rescue / force-EOS / final <c>NucleusSampler.Draw</c> call) is untouched by the fix
    /// and reproduced here unchanged so the full method's behavior — not just the sort — is compared.</summary>
    private static int ReferenceSample(float[] cond, float[] guided, int channel, DiaConfig cfg, ref uint rng,
        int topK, float temperature, float topP)
    {
        int v = cfg.AudioVocab;
        int k = Math.Min(topK, v);
        int[] order = new int[v];
        for (int i = 0; i < v; i++) order[i] = i;
        Array.Sort(order, (a, b) => guided[b].CompareTo(guided[a]));
        float[] arr = new float[v];
        Array.Fill(arr, float.NegativeInfinity);
        for (int r = 0; r < k; r++) arr[order[r]] = cond[order[r]];
        if (channel == 0 && float.IsNegativeInfinity(arr[cfg.AudioEos]))
        {
            float eosCond = cond[cfg.AudioEos];
            int rank = 0;
            for (int i = 0; i <= cfg.AudioEos && rank < k; i++) if (cond[i] > eosCond) rank++;
            if (rank < k) arr[cfg.AudioEos] = eosCond;
        }
        for (int i = cfg.AudioEos + 1; i < v; i++) arr[i] = float.NegativeInfinity;
        if (channel != 0) arr[cfg.AudioEos] = float.NegativeInfinity;
        int top = 0;
        for (int i = 1; i < v; i++) if (arr[i] > arr[top]) top = i;
        if (top == cfg.AudioEos)
        {
            for (int i = 0; i < cfg.AudioEos; i++) arr[i] = float.NegativeInfinity;
        }
        else
        {
            arr[cfg.AudioEos] = float.NegativeInfinity;
        }
        return HartsyInference.Audio.Sampling.NucleusSampler.Draw(arr, v, temperature, 0, topP, ref rng);
    }

    private static float[] MakeLogits(int n, Random rng, float scale = 10.0f)
    {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * scale * 2 - scale);
        return v;
    }

    private static void AssertIdentical(float[] cond, float[] guided, int channel, int topK, float temperature,
        float topP, int seed, string label)
    {
        uint rngOld = DeterministicRng.Seed(seed);
        uint rngNew = DeterministicRng.Seed(seed);
        int tokenOld = ReferenceSample((float[])cond.Clone(), (float[])guided.Clone(), channel, Cfg, ref rngOld, topK, temperature, topP);
        int tokenNew = DiaPipeline.SampleDiaChannel((float[])cond.Clone(), (float[])guided.Clone(), channel, Cfg, ref rngNew, topK, temperature, topP);
        Assert.True(tokenOld == tokenNew, $"{label}: old={tokenOld} new={tokenNew}");
        Assert.Equal(rngOld, rngNew);
    }

    [Theory]
    [InlineData(0)] // channel 0 carries EOS
    [InlineData(1)]
    [InlineData(8)] // last of the 9 DAC channels
    public void RandomLogits_MatchesReference(int channel)
    {
        Random rng = new(55_000 + channel);
        for (int trial = 0; trial < 20; trial++)
        {
            float[] cond = MakeLogits(Cfg.AudioVocab, rng);
            float[] guided = MakeLogits(Cfg.AudioVocab, rng);
            AssertIdentical(cond, guided, channel, Cfg.TopK, Cfg.Temperature, Cfg.TopP,
                seed: 1 + trial, label: $"channel={channel} trial={trial}");
        }
    }

    /// <summary>Exact ties at the cfg.TopK window boundary in GUIDED (the array the window is selected from,
    /// not the one sampled from) -- the one scenario where a tie-break difference in the window-selection sort
    /// could change which conditional value enters the candidate set.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(10)] // more ties than fit in the TopK=45 window
    public void ExactTiesAtWindowBoundary_MatchesReference(int numTied)
    {
        Random rng = new(66_000 + numTied);
        float[] cond = MakeLogits(Cfg.AudioVocab, rng);
        float[] guided = MakeLogits(Cfg.AudioVocab, rng);
        // Place `numTied` identical values straddling the TopK-th rank of `guided` so some of them are the
        // deciding tie for window membership.
        float boundaryValue = 3.0f;
        for (int i = 0; i < numTied; i++) guided[(Cfg.TopK - numTied / 2 + i + Cfg.AudioVocab) % Cfg.AudioVocab] = boundaryValue;
        for (int trial = 0; trial < 10; trial++)
        {
            AssertIdentical(cond, guided, channel: 1, Cfg.TopK, Cfg.Temperature, Cfg.TopP,
                seed: 200 + trial, label: $"numTied={numTied} trial={trial}");
        }
    }

    /// <summary>Channel 0's EOS-rescue path specifically: force the conditional's EOS logit high enough to
    /// rank inside the window so the rescue branch fires, and also exercise the force-EOS argmax branch.</summary>
    [Fact]
    public void Channel0_EosRescueAndForceEos_MatchesReference()
    {
        Random rng = new(77_000);
        float[] cond = MakeLogits(Cfg.AudioVocab, rng);
        float[] guided = MakeLogits(Cfg.AudioVocab, rng);
        cond[Cfg.AudioEos] = 100f;     // conditional strongly prefers EOS
        guided[Cfg.AudioEos] = -100f;  // guided (CFG-combined) does NOT rank EOS in its top-K
        for (int trial = 0; trial < 10; trial++)
        {
            AssertIdentical(cond, guided, channel: 0, Cfg.TopK, Cfg.Temperature, Cfg.TopP,
                seed: 300 + trial, label: $"eos-rescue trial={trial}");
        }
    }
}
