using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Numerical parity for the rewritten <see cref="CausalConditionalDecoder"/> (CosyVoice 2 CFM
/// estimator) against the real <c>s3gen.safetensors</c> <c>flow.decoder.estimator.*</c>. Feeds the fixed
/// PyTorch-reference inputs (<c>ref_decoder_io.safetensors</c>: x/mu/spks/cond/t) and diffs the velocity
/// output. Gated on <c>S3GEN_PATH</c> + <c>REF_DECODER_IO</c>.</summary>
public sealed unsafe class CosyVoiceDecoderParityTests
{
    private readonly ITestOutputHelper _out;
    public CosyVoiceDecoderParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void Decoder_MatchesPythonReference()
    {
        string? s3gen = Environment.GetEnvironmentVariable("S3GEN_PATH");
        string? refIo = Environment.GetEnvironmentVariable("REF_DECODER_IO");
        if (string.IsNullOrEmpty(s3gen) || !File.Exists(s3gen) || string.IsNullOrEmpty(refIo) || !File.Exists(refIo))
            return; // gated

        SafeTensorsLoader sl = new();
        sl.Load(s3gen);
        Dictionary<string, Tensor> decW = new();
        const string pre = "flow.decoder.estimator.";
        foreach (KeyValuePair<string, Tensor> kv in sl.GetAllTensors())
            if (kv.Key.StartsWith(pre, StringComparison.Ordinal)) decW[kv.Key[pre.Length..]] = kv.Value;
        _out.WriteLine($"flow.decoder.estimator.* tensors: {decW.Count}");

        CausalConditionalDecoder dec = new(CosyVoiceConfig.V2_0_5B.Flow);
        dec.LoadWeights(decW, prefix: "");

        SafeTensorsLoader io = new();
        io.Load(refIo);
        IReadOnlyDictionary<string, Tensor> d = io.GetAllTensors();
        Tensor x = d["x"], mu = d["mu"], spks = d["spks"], cond = d["cond"], refOut = d["dec_out"];
        float t = ((float*)d["t"].DataPointer)[0];

        using CpuBackend backend = new();
        using Tensor outT = dec.Estimate(backend, x, mu, t, spks, cond);

        Assert.Equal(refOut.Shape[1], outT.Shape[1]);
        Assert.Equal(refOut.Shape[2], outT.Shape[2]);
        float* a = (float*)outT.DataPointer;
        float* b = (float*)refOut.DataPointer;
        long n = outT.ElementCount;
        double maxAbs = 0, refPeak = 0;
        for (long i = 0; i < n; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(a[i] - b[i]));
            refPeak = Math.Max(refPeak, Math.Abs(b[i]));
        }
        _out.WriteLine($"out [{outT.Shape[1]},{outT.Shape[2]}] t={t}. maxAbs={maxAbs:E4} (ref peak {refPeak:F4}).");
        _out.WriteLine($"  mine[0,0,:6]=[{a[0]:F5},{a[1]:F5},{a[2]:F5},{a[3]:F5},{a[4]:F5},{a[5]:F5}]");
        _out.WriteLine($"  ref [0,0,:6]=[{b[0]:F5},{b[1]:F5},{b[2]:F5},{b[3]:F5},{b[4]:F5},{b[5]:F5}]");
        Assert.True(maxAbs < 5e-3, $"decoder output diverges from reference (maxAbs={maxAbs:E4}).");
    }
}
