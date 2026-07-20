using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Chatterbox;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Numerical parity for <see cref="ChatterboxPerceiver"/> (T3 cond-prompt resampler) against the real
/// <c>t3_cfg.safetensors</c> <c>cond_enc.perceiver.*</c>. Feeds the fixed PyTorch-reference prompt embeddings
/// and diffs the 32 resampled conditioning tokens. Gated on <c>T3_PATH</c> + <c>REF_PERCEIVER_IO</c>.</summary>
public sealed unsafe class ChatterboxPerceiverParityTests
{
    private readonly ITestOutputHelper _out;
    public ChatterboxPerceiverParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void Perceiver_MatchesPythonReference()
    {
        string? t3 = Environment.GetEnvironmentVariable("T3_PATH");
        string? refIo = Environment.GetEnvironmentVariable("REF_PERCEIVER_IO");
        if (string.IsNullOrEmpty(t3) || !File.Exists(t3) || string.IsNullOrEmpty(refIo) || !File.Exists(refIo))
            return; // gated

        SafeTensorsLoader sl = new();
        sl.Load(t3);
        Dictionary<string, Tensor> w = new();
        foreach (KeyValuePair<string, Tensor> kv in sl.GetAllTensors())
            if (kv.Key.StartsWith("cond_enc.perceiver.", StringComparison.Ordinal)) w[kv.Key] = kv.Value;

        using ChatterboxPerceiver perc = new();
        perc.LoadWeights(w);

        SafeTensorsLoader io = new();
        io.Load(refIo);
        IReadOnlyDictionary<string, Tensor> d = io.GetAllTensors();
        Tensor pin = d["perc_in"];     // [1,150,1024]
        Tensor refOut = d["perc_out"]; // [1,32,1024]

        using CpuBackend backend = new();
        using Tensor outT = perc.Forward(backend, pin);

        Assert.Equal(refOut.Shape[1], outT.Shape[1]);
        Assert.Equal(refOut.Shape[2], outT.Shape[2]);
        float* a = (float*)outT.DataPointer;
        float* b = (float*)refOut.DataPointer;
        long n = outT.ElementCount;
        double maxAbs = 0, refPeak = 0;
        for (long i = 0; i < n; i++) { maxAbs = Math.Max(maxAbs, Math.Abs(a[i] - b[i])); refPeak = Math.Max(refPeak, Math.Abs(b[i])); }
        _out.WriteLine($"out [{outT.Shape[1]},{outT.Shape[2]}]. maxAbs={maxAbs:E4} (ref peak {refPeak:F4}).");
        _out.WriteLine($"  mine[0,0,:5]=[{a[0]:F5},{a[1]:F5},{a[2]:F5},{a[3]:F5},{a[4]:F5}]  ref=[{b[0]:F5},{b[1]:F5},{b[2]:F5},{b[3]:F5},{b[4]:F5}]");
        GC.KeepAlive(sl);
        Assert.True(maxAbs < 2e-3, $"perceiver diverges (maxAbs={maxAbs:E4}).");
        sl.Dispose();
    }
}
