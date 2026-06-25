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

/// <summary>Numerical parity for the Kyutai/moshi <see cref="MoshiConditioner"/> (cfg/control LUT sum + speaker
/// cross source with sinusoidal pos-emb) against the real checkpoint. The Python reference builds the condition
/// tensors for a fixed voice + cfg 2.0 + control "ok" and dumps the fuser's sum/cross. Gated on
/// <c>KYUTAI_TTS_WEIGHTS</c> + <c>KYUTAI_REF_CONDITIONER</c>.</summary>
public sealed unsafe class KyutaiConditionerParityTests
{
    private readonly ITestOutputHelper _out;
    public KyutaiConditionerParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void Conditioner_MatchesMoshiReference()
    {
        string? wp = Environment.GetEnvironmentVariable("KYUTAI_TTS_WEIGHTS");
        string? rp = Environment.GetEnvironmentVariable("KYUTAI_REF_CONDITIONER");
        if (string.IsNullOrEmpty(wp) || !File.Exists(wp) || string.IsNullOrEmpty(rp) || !File.Exists(rp)) return;

        using SafeTensorsLoader weights = new(); weights.Load(wp);
        using SafeTensorsLoader io = new(); io.Load(rp);
        IReadOnlyDictionary<string, Tensor> w = weights.GetAllTensors();
        IReadOnlyDictionary<string, Tensor> d = io.GetAllTensors();

        Tensor voice = d["voice"];          // [1,T,512]
        Tensor refSum = d["sum_cond"];      // [1,1,2048]
        Tensor refCross = d["cross"];       // [1,T,2048]
        int t = (int)voice.Shape[1];

        using MoshiConditioner cond = new();
        cond.LoadWeights(w);
        using CpuBackend backend = new();
        using Tensor sum = cond.ComputeSum(backend, MoshiConditioner.CfgBin(2.0f));
        using Tensor cross = cond.ComputeCross(backend, voice);

        double sumMax = MaxAbs((float*)sum.DataPointer, (float*)refSum.DataPointer, MoshiConditioner.Dim);
        double crossMax = MaxAbs((float*)cross.DataPointer, (float*)refCross.DataPointer, (long)t * MoshiConditioner.Dim);
        _out.WriteLine($"cfgBin={MoshiConditioner.CfgBin(2.0f)}; sum maxAbs={sumMax:E4}; cross[1,{t},2048] maxAbs={crossMax:E4}.");
        _out.WriteLine($"  sum mine[:4]=[{((float*)sum.DataPointer)[0]:F5},{((float*)sum.DataPointer)[1]:F5},{((float*)sum.DataPointer)[2]:F5},{((float*)sum.DataPointer)[3]:F5}]");
        _out.WriteLine($"  sum ref [:4]=[{((float*)refSum.DataPointer)[0]:F5},{((float*)refSum.DataPointer)[1]:F5},{((float*)refSum.DataPointer)[2]:F5},{((float*)refSum.DataPointer)[3]:F5}]");
        Assert.True(sumMax < 5e-3, $"sum diverges ({sumMax:E4}).");
        Assert.True(crossMax < 5e-3, $"cross diverges ({crossMax:E4}).");
    }

    private static double MaxAbs(float* a, float* b, long n)
    {
        double m = 0;
        for (long i = 0; i < n; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }
}
