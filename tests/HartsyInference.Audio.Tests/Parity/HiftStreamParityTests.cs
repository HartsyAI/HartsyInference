using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Cuda;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Self-parity for CosyVoice2's HiFTNet vocoder streaming path: a monolithic
/// <see cref="HiFTNetVocoder.Forward"/> call over a whole synthetic mel must reconstruct (to float rounding)
/// the same PCM as <see cref="HiFTNetVocoder.ForwardStreaming"/> called chunk-by-chunk with a carried
/// <see cref="HiFTStreamState"/>. Covers chunk sizes down to 1 mel frame (the strongest cross-boundary case)
/// up to the whole utterance in one call (no boundary at all — the baseline every other size is compared
/// against). <c>HIFT_DETERMINISTIC=1</c> kills the NSF source's stochastic noise so the two paths are
/// comparable regardless of noise-RNG call-count differences (see <see cref="HiFTNetVocoder.Forward"/>).
/// Gated on <c>COSYVOICE2_HIFT_WEIGHTS</c> (the FunAudioLLM/CosyVoice2-0.5B <c>hift.pt</c>, already on disk
/// from earlier CosyVoice2 work) so this runs against real weights, not synthetic ones. CUDA-only (matches
/// how <see cref="HiFTNetVocoder"/> is actually driven in production — the DSP steps are host-side either
/// way, but the convs/resblocks are backend-dispatched).
///
/// <para><b>KNOWN LIMITATION — this test currently passes for the WRONG reason and does not gate real
/// correctness.</b> <see cref="HiFTNetVocoder.ForwardStreaming"/>'s "recompute Forward() over the growing
/// mel-so-far, emit only the settled suffix" design (see <see cref="HiFTStreamState"/>'s doc comment) is
/// exact for the local/convolutional parts of the network, but the NSF harmonic source's phase accumulator
/// (<c>cum[h]</c> in <c>NsfVocoderDsp.GenerateHarmonicSource</c>) is an UNBOUNDED RUNNING SUM — tiny
/// per-call numeric differences at "settled" F0-predictor positions (same order as the ~1e-4 shape-dependent
/// GPU noise this design already tolerates elsewhere) get integrated into that sum and, inside a
/// <c>sin()</c>, do not decay with distance from the current edge the way local conv error does. Verified
/// directly 2026-08-10: a 120-frame synthetic mel (this test's current length) stays under the margin needed
/// to hide this, so it passes trivially — but extending <c>totalMelFrames</c> to 600 (~12s, still synthetic
/// RANDOM mel, nothing speech-specific) reproduces relL2≈0.63 regardless of chunk size, and the real
/// CosyVoice2-0.5B end-to-end mel (9.36s) shows the same failure at margin=48k (relL2≈0.73) that only
/// shrinks to ~0 once the margin approaches the ENTIRE utterance length — i.e. once no real streaming is
/// happening at all. This confirms the ORIGINAL plan's own risk call
/// (<c>~/.claude/plans/snoopy-napping-meadow.md</c> §5.1): the harmonic source needs REAL CARRIED phase
/// state, not recompute-with-margin. <see cref="HiFTNetVocoder.ForwardStreaming"/> is NOT production-ready
/// and must not be wired into <c>CosyVoicePipeline</c> or deployed until that redesign lands — this test's
/// short synthetic length is a known gap, not a clean bar, until then.</para></summary>
public sealed unsafe class HiftStreamParityTests
{
    private readonly ITestOutputHelper _out;
    public HiftStreamParityTests(ITestOutputHelper o) => _out = o;

    [Theory]
    [Trait("Category", "Integration")]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(6)]
    [InlineData(12)]
    [InlineData(200)] // "whole" — larger than the synthetic mel below, so this is a single one-shot call
    public void StreamingForward_MatchesMonolithicForward(int chunkFrames)
    {
        string? wPath = Environment.GetEnvironmentVariable("COSYVOICE2_HIFT_WEIGHTS");
        if (string.IsNullOrEmpty(wPath) || !File.Exists(wPath))
        {
            return; // gated, same convention as MimiStreamParityTests
        }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _out.WriteLine($"SKIPPED: PTX directory not found at {ptxDir}.");
            return;
        }
        Environment.SetEnvironmentVariable("HIFT_DETERMINISTIC", "1");

        PytorchPickleLoader loader = new();
        loader.Load(wPath);
        IReadOnlyDictionary<string, Tensor> w = loader.GetAllTensors();

        CosyVoiceHiftConfig cfg = new();
        HiFTNetVocoder mono = new(cfg);
        mono.LoadWeights(w);
        HiFTNetVocoder stream = new(cfg);
        stream.LoadWeights(w);

        const int totalMelFrames = 120; // ~2.4s at the 50 Hz mel rate — see the KNOWN LIMITATION note above
        Random rng = new(1234);
        Tensor mel = new(new TensorShape(1, cfg.MelBins, totalMelFrames), DType.F32);
        float* mp = (float*)mel.DataPointer;
        for (int c = 0; c < cfg.MelBins; c++)
            for (int t = 0; t < totalMelFrames; t++)
                mp[c * totalMelFrames + t] = (float)(rng.NextDouble() * 4.0 - 6.0); // plausible log-mel range

        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);

        float[] reference = mono.Forward(backend, mel);

        using HiFTStreamState state = new();
        List<float> candidate = new(reference.Length);
        for (int start = 0; start < totalMelFrames; start += chunkFrames)
        {
            int n = Math.Min(chunkFrames, totalMelFrames - start);
            bool isFinal = start + n >= totalMelFrames;
            Tensor chunkMel = new(new TensorShape(1, cfg.MelBins, n), DType.F32);
            float* ccp = (float*)chunkMel.DataPointer;
            for (int c = 0; c < cfg.MelBins; c++)
                for (int t = 0; t < n; t++)
                    ccp[c * n + t] = mp[c * totalMelFrames + (start + t)];

            float[] chunkAudio = stream.ForwardStreaming(backend, chunkMel, state, isFinal);
            chunkMel.Dispose();
            candidate.AddRange(chunkAudio);
        }
        mel.Dispose();

        Assert.Equal(reference.Length, candidate.Count);

        double maxAbs = 0, sumSq = 0, refSumSq = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double diff = candidate[i] - reference[i];
            if (Math.Abs(diff) > maxAbs) maxAbs = Math.Abs(diff);
            sumSq += diff * diff;
            refSumSq += reference[i] * (double)reference[i];
        }
        double relL2 = Math.Sqrt(sumSq / Math.Max(refSumSq, 1e-12));
        _out.WriteLine($"chunkFrames={chunkFrames} samples={reference.Length} maxAbs={maxAbs:E4} relL2={relL2:E4}");

        Assert.True(maxAbs < 1e-3, $"streaming vs monolithic maxAbs too high: {maxAbs:E4}");
        Assert.True(relL2 < 1e-3, $"streaming vs monolithic relL2 too high: {relL2:E4}");
    }
}
