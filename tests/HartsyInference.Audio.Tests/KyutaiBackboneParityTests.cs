using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Kyutai;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Numerical parity for the Kyutai/moshi temporal backbone (<see cref="MoshiTransformer"/>) against the
/// real <c>kyutai/tts-1.6b-en_fr</c> checkpoint. The Python reference (moshi 0.2.13) runs
/// <c>lm.out_norm(lm.transformer(input, cross_attention_src=cross))</c> on fixed tensors and saves IO to
/// <c>ref_backbone_io.safetensors</c>; this loads the same weights, runs the C# backbone, and diffs the result.
///
/// <para>Gated on <c>KYUTAI_TTS_WEIGHTS</c> (dsm_tts safetensors) + <c>KYUTAI_REF_BACKBONE</c> (the dumped IO).</para></summary>
public sealed unsafe class KyutaiBackboneParityTests
{
    private readonly ITestOutputHelper _out;
    public KyutaiBackboneParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void Backbone_MatchesMoshiReference()
    {
        string? wp = Environment.GetEnvironmentVariable("KYUTAI_TTS_WEIGHTS");
        string? rp = Environment.GetEnvironmentVariable("KYUTAI_REF_BACKBONE");
        if (string.IsNullOrEmpty(wp) || !File.Exists(wp) || string.IsNullOrEmpty(rp) || !File.Exists(rp)) return;

        using SafeTensorsLoader weights = new(); weights.Load(wp);
        using SafeTensorsLoader io = new(); io.Load(rp);
        IReadOnlyDictionary<string, Tensor> w = weights.GetAllTensors();
        IReadOnlyDictionary<string, Tensor> d = io.GetAllTensors();

        Tensor input = d["input"];     // [1, T, 2048]
        Tensor cross = d["cross"];     // [1, S, 2048]
        Tensor refOut = d["out_norm"]; // [1, T, 2048]
        int t = (int)input.Shape[1], s = (int)cross.Shape[1];

        using MoshiTransformer bb = new(layers: 16);
        bb.LoadWeights(w);
        using CpuBackend backend = new();
        using Tensor outT = bb.Forward(backend, input, cross);

        Assert.Equal((long)t, outT.Shape[1]);
        float* a = (float*)outT.DataPointer; float* b = (float*)refOut.DataPointer;
        double maxAbs = 0, peak = 0;
        for (long i = 0; i < (long)t * MoshiTransformer.Dim; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(a[i] - b[i]));
            peak = Math.Max(peak, Math.Abs(b[i]));
        }
        _out.WriteLine($"backbone [1,{t},2048] vs ref. maxAbs={maxAbs:E4} (ref peak {peak:F4}).");
        _out.WriteLine($"  mine[:5]=[{a[0]:F5},{a[1]:F5},{a[2]:F5},{a[3]:F5},{a[4]:F5}]");
        _out.WriteLine($"  ref [:5]=[{b[0]:F5},{b[1]:F5},{b[2]:F5},{b[3]:F5},{b[4]:F5}]");
        Assert.True(maxAbs < 5e-3, $"backbone diverges (maxAbs={maxAbs:E4}).");
    }
}
