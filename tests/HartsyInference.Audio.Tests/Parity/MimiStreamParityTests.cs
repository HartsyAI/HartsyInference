using System;
using System.Collections.Generic;
using HartsyInference.Audio.Models.Codecs.Mimi;
using MimiModel = HartsyInference.Audio.Models.Codecs.Mimi.Mimi;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Self-parity for Mimi's streaming decode: a monolithic <see cref="MimiModel.Decode"/> call over a
/// whole utterance must reconstruct (to float rounding) the same PCM as <see cref="MimiModel.DecodeStreaming"/>
/// called chunk-by-chunk with a carried <see cref="MimiDecoderStreamState"/>. Covers chunk sizes down to 1 frame
/// (chunkFrames=1 matches kyutai's own reference client's emission granularity, the strongest cross-boundary
/// case) up to 24 (the whole utterance in one call, i.e. no boundary at all — the baseline every other size is
/// compared against). Gated on <c>MIMI_DSM_WEIGHTS</c> (the kyutai/tts-1.6b-en_fr Mimi tokenizer safetensors,
/// already on disk from the Kyutai TTS download) so this runs against real weights, not synthetic ones — real
/// weights are what actually exercise the RVQ/transformer/SEANet math this change touches. This is the regression
/// guard for every model that reuses this codec (e.g. CSM), not just Kyutai.</summary>
public sealed unsafe class MimiStreamParityTests
{
    private readonly ITestOutputHelper _out;
    public MimiStreamParityTests(ITestOutputHelper o) => _out = o;

    [Theory]
    [Trait("Category", "Integration")]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(6)]
    [InlineData(12)]
    [InlineData(24)]
    public void StreamingDecode_MatchesMonolithicDecode(int chunkFrames)
    {
        string? wPath = Environment.GetEnvironmentVariable("MIMI_DSM_WEIGHTS");
        if (string.IsNullOrEmpty(wPath) || !System.IO.File.Exists(wPath))
        {
            return; // gated, same convention as MimiRvqParityTests
        }

        SafeTensorsLoader wl = new(); wl.Load(wPath);
        IReadOnlyDictionary<string, Tensor> w = wl.GetAllTensors();

        MimiModel mono = new(MimiConfig.Mimi24kHzDsm);
        mono.LoadWeights(w);
        MimiModel stream = new(MimiConfig.Mimi24kHzDsm);
        stream.LoadWeights(w);

        const int totalFrames = 24;
        const int codebooks = 32; // Mimi24kHzDsm.TotalCodebooks
        Random rng = new(1234);
        Tensor codes = new(new TensorShape(1, codebooks, totalFrames), DType.I32);
        int* cp = (int*)codes.DataPointer;
        for (int k = 0; k < codebooks; k++)
            for (int f = 0; f < totalFrames; f++)
                cp[k * totalFrames + f] = rng.Next(0, 2048);

        using CpuBackend backend = new();

        Tensor reference = mono.Decode(backend, codes, batch: 1, tFrames: totalFrames);
        int refSamples = (int)reference.Shape[2];

        using MimiDecoderStreamState state = new();
        float[] candidate = new float[refSamples];
        int sampleCursor = 0;
        for (int start = 0; start < totalFrames; start += chunkFrames)
        {
            int n = Math.Min(chunkFrames, totalFrames - start);
            Tensor chunkCodes = new(new TensorShape(1, codebooks, n), DType.I32);
            int* ccp = (int*)chunkCodes.DataPointer;
            for (int k = 0; k < codebooks; k++)
                for (int f = 0; f < n; f++)
                    ccp[k * n + f] = cp[k * totalFrames + (start + f)];

            Tensor chunkPcm = stream.DecodeStreaming(backend, chunkCodes, batch: 1, tFrames: n, state);
            chunkCodes.Dispose();
            int chunkSamples = (int)chunkPcm.Shape[2];
            float* src = (float*)chunkPcm.DataPointer;
            for (int i = 0; i < chunkSamples; i++) candidate[sampleCursor + i] = src[i];
            sampleCursor += chunkSamples;
            chunkPcm.Dispose();
        }
        Assert.Equal(refSamples, sampleCursor);

        float* refP = (float*)reference.DataPointer;
        double maxAbs = 0, sumSq = 0, refSumSq = 0;
        for (int i = 0; i < refSamples; i++)
        {
            double diff = candidate[i] - refP[i];
            if (Math.Abs(diff) > maxAbs) maxAbs = Math.Abs(diff);
            sumSq += diff * diff;
            refSumSq += refP[i] * (double)refP[i];
        }
        double relL2 = Math.Sqrt(sumSq / Math.Max(refSumSq, 1e-12));
        _out.WriteLine($"chunkFrames={chunkFrames} samples={refSamples} maxAbs={maxAbs:E4} relL2={relL2:E4}");

        codes.Dispose();
        reference.Dispose();

        Assert.True(maxAbs < 1e-3, $"streaming vs monolithic maxAbs too high: {maxAbs:E4}");
        Assert.True(relL2 < 1e-3, $"streaming vs monolithic relL2 too high: {relL2:E4}");
    }
}
