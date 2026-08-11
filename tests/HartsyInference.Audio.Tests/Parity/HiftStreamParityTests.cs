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

/// <summary>Self-parity for CosyVoice2's HiFTNet vocoder streaming path: <see cref="HiFTNetVocoder.ForwardStreaming"/>
/// called chunk-by-chunk with a carried <see cref="HiFTStreamState"/> must reconstruct the same PCM as a
/// monolithic call over the whole utterance at once. The reference is <see cref="HiFTNetVocoder.ForwardHostCpuF0"/>,
/// NOT the plain <see cref="HiFTNetVocoder.Forward"/> — comparing against <c>Forward</c> would conflate two
/// different questions ("is the chunking logic exact" vs. "does host-CPU F0Predictor match GPU F0Predictor's
/// cuDNN rounding", the latter a real but separate ~2e-3-level relL2 gap, see below). Covers chunk sizes down
/// to 1 mel frame (the strongest cross-boundary case) up to the whole utterance in one call (no boundary at
/// all). <c>HIFT_DETERMINISTIC=1</c> kills the NSF source's stochastic noise so the two paths are comparable
/// regardless of noise-RNG call-count differences. Gated on <c>COSYVOICE2_HIFT_WEIGHTS</c> (the
/// FunAudioLLM/CosyVoice2-0.5B <c>hift.pt</c>) so this runs against real weights.
///
/// <para><b>Design history, worth keeping</b> (see <see cref="HiFTStreamState"/>'s doc comment for the full
/// account): the first implementation recomputed the harmonic source's phase from t=0 every call — provably
/// wrong, since the NSF phase accumulator is an unbounded running sum and even tiny per-call GPU numeric
/// noise compounds with utterance length instead of decaying (confirmed empirically: relL2≈0.6-0.8 on
/// utterances beyond a few seconds, converging only once the margin approached the whole utterance, i.e. no
/// real streaming). The fix carries phase/RNG state and consumes each historical F0 value exactly once. That
/// alone wasn't sufficient either — F0Predictor's OWN <c>backend.Conv1d</c> (cuDNN) output for a fixed
/// INTERIOR position varies by ~0.01-0.16 Hz depending on the window's total length (shape-dependent
/// algorithm selection, not boundary effects), so "settled by margin" never held for it. The real fix is
/// <see cref="F0Predictor.ForwardHostCpu"/>, a naive host reimplementation with no such shape-dependent path.
/// With both fixes: chunked vs. a matched host-CPU-F0 monolithic reference is exact to relL2≈2e-3 REGARDLESS
/// of utterance length (verified at both 60 and 600 synthetic mel frames, and against the real 9.36s
/// CosyVoice2 end-to-end mel) — a bounded, non-growing floor from the x-path's OWN cuDNN shape-dependent
/// noise (same class of issue as F0Predictor's, just far smaller since nothing here feeds an accumulator),
/// not a correctness bug. See the real-generation listen test referenced in
/// <c>audiolab-held-items-2026-08-10.md</c> for the perceptual verification this numeric tolerance rests on.</para></summary>
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
    [InlineData(600)] // "whole" — larger than the synthetic mel below, so this is a single one-shot call
    public void StreamingForward_MatchesHostCpuF0MonolithicForward(int chunkFrames)
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

        // 600 frames (~12s) — long enough to exercise genuine incremental settlement across ~100 calls at
        // chunkFrames=6, not just the trivial "everything via the final isFinal flush" path a short mel would
        // hide behind (this is exactly the gap that let the original broken design pass its own first test).
        const int totalMelFrames = 600;
        Random rng = new(1234);
        Tensor mel = new(new TensorShape(1, cfg.MelBins, totalMelFrames), DType.F32);
        float* mp = (float*)mel.DataPointer;
        for (int c = 0; c < cfg.MelBins; c++)
            for (int t = 0; t < totalMelFrames; t++)
                mp[c * totalMelFrames + t] = (float)(rng.NextDouble() * 4.0 - 6.0); // plausible log-mel range

        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);

        Tensor monoMel = CopyMel(mel);
        float[] reference = mono.ForwardHostCpuF0(backend, monoMel);
        monoMel.Dispose();

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

        // 5e-3, not 1e-3: the measured floor (relL2≈1.9-2.3e-3 across chunk sizes and margins 48k-96k) is a
        // bounded, non-growing GPU cuDNN shape-dependent noise floor in the x-path's own convs (see class doc
        // comment) — real, understood, and NOT the utterance-length-scaling bug this test exists to catch.
        Assert.True(relL2 < 5e-3, $"streaming vs hostCPU-F0 monolithic relL2 too high: {relL2:E4}");
    }

    private static Tensor CopyMel(Tensor mel)
    {
        Tensor copy = new(mel.Shape, DType.F32);
        Buffer.MemoryCopy((void*)mel.DataPointer, (void*)copy.DataPointer, mel.ElementCount * 4, mel.ElementCount * 4);
        return copy;
    }
}
