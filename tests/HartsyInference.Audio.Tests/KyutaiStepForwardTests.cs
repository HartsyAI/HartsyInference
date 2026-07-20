using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Kyutai;
using HartsyInference.Cpu;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Isolates the streaming KV-cache path: runs <see cref="MoshiTransformer.StepForward"/> one token at a
/// time (with a <see cref="FixedKvCache"/> + precomputed cross K/V) over the same fixed input the full-sequence
/// <see cref="MoshiTransformer.Forward"/> was validated against, and requires the per-position output to match
/// the reference <c>out_norm</c> (which Forward hits at ~1.3e-4). A gap here means the KV-cache step diverges
/// from the validated full-prefix path — the prime suspect for greedy-generation argmax flips. Gated on
/// <c>KYUTAI_TTS_WEIGHTS</c> + <c>KYUTAI_REF_BACKBONE</c>.</summary>
public sealed unsafe class KyutaiStepForwardTests
{
    private readonly ITestOutputHelper _out;
    public KyutaiStepForwardTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void StepForward_MatchesFullPrefixReference()
    {
        string? wp = Environment.GetEnvironmentVariable("KYUTAI_TTS_WEIGHTS");
        string? rp = Environment.GetEnvironmentVariable("KYUTAI_REF_BACKBONE");
        if (string.IsNullOrEmpty(wp) || !File.Exists(wp) || string.IsNullOrEmpty(rp) || !File.Exists(rp)) return;

        using SafeTensorsLoader weights = new(); weights.Load(wp);
        using SafeTensorsLoader io = new(); io.Load(rp);
        IReadOnlyDictionary<string, Tensor> w = weights.GetAllTensors();
        IReadOnlyDictionary<string, Tensor> d = io.GetAllTensors();

        Tensor input = d["input"];     // [1,T,2048]
        Tensor cross = d["cross"];     // [1,S,2048]
        Tensor refOut = d["out_norm"]; // [1,T,2048]
        int t = (int)input.Shape[1];

        using MoshiTransformer bb = new(layers: 16);
        bb.LoadWeights(w);
        using CpuBackend backend = new();

        using MoshiTransformer.CrossKvCache crossKv = bb.PrecomputeCrossKv(backend, cross);
        using FixedKvCache selfCache = new(numLayers: 16, batch: 1, numKvHeads: MoshiTransformer.Heads,
            headDim: MoshiTransformer.HeadDim, maxSequenceLength: t);

        float* ip = (float*)input.DataPointer;
        float* rp2 = (float*)refOut.DataPointer;
        double maxAbs = 0;
        for (int pos = 0; pos < t; pos++)
        {
            Tensor frame = new(new TensorShape(1, 1, MoshiTransformer.Dim), DType.F32);
            Buffer.MemoryCopy(ip + (long)pos * MoshiTransformer.Dim, (void*)frame.DataPointer, (long)MoshiTransformer.Dim * 4, (long)MoshiTransformer.Dim * 4);
            using Tensor outT = bb.StepForward(backend, frame, crossKv, selfCache, pos);
            frame.Dispose();
            float* op = (float*)outT.DataPointer;
            double rowMax = 0;
            for (int i = 0; i < MoshiTransformer.Dim; i++)
                rowMax = Math.Max(rowMax, Math.Abs(op[i] - rp2[(long)pos * MoshiTransformer.Dim + i]));
            maxAbs = Math.Max(maxAbs, rowMax);
            _out.WriteLine($"pos {pos}: rowMax={rowMax:E4}");
        }
        _out.WriteLine($"StepForward vs full-prefix reference: maxAbs={maxAbs:E4}.");
        Assert.True(maxAbs < 5e-3, $"StepForward diverges from the validated full-prefix path ({maxAbs:E4}).");
    }
}
