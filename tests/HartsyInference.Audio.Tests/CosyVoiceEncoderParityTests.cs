using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Numerical parity for the rewritten <see cref="UpsampleConformerEncoder"/> (CosyVoice 2 flow token
/// encoder) against the real ResembleAI/chatterbox <c>s3gen.safetensors</c>. Loads <c>flow.encoder.*</c>, runs
/// a fixed input, and diffs against the PyTorch reference dump (<c>ref_encoder_io.safetensors</c>:
/// <c>enc_in</c> [1,T,512] → <c>enc_out</c> [1,2T,512]).
///
/// <para>Gated on <c>S3GEN_PATH</c> (the real s3gen.safetensors) + <c>REF_ENCODER_IO</c> (the dump). Tolerance
/// is F32-accumulation level.</para></summary>
public sealed unsafe class CosyVoiceEncoderParityTests
{
    private readonly ITestOutputHelper _out;
    public CosyVoiceEncoderParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void Encoder_MatchesPythonReference()
    {
        string? s3gen = Environment.GetEnvironmentVariable("S3GEN_PATH");
        string? refIo = Environment.GetEnvironmentVariable("REF_ENCODER_IO");
        if (string.IsNullOrEmpty(s3gen) || !File.Exists(s3gen) || string.IsNullOrEmpty(refIo) || !File.Exists(refIo))
            return; // gated

        SafeTensorsLoader sl = new();
        sl.Load(s3gen);
        Dictionary<string, Tensor> encW = new();
        const string pre = "flow.encoder.";
        foreach (KeyValuePair<string, Tensor> kv in sl.GetAllTensors())
            if (kv.Key.StartsWith(pre, StringComparison.Ordinal)) encW[kv.Key[pre.Length..]] = kv.Value;
        _out.WriteLine($"flow.encoder.* tensors: {encW.Count}");

        using UpsampleConformerEncoder enc = new(outputSize: 512, numHeads: 8, numPreBlocks: 6, numPostBlocks: 4);
        enc.LoadWeights(encW);

        SafeTensorsLoader io = new();
        io.Load(refIo);
        IReadOnlyDictionary<string, Tensor> iod = io.GetAllTensors();
        Tensor encIn = iod["enc_in"];     // [1, T, 512]
        Tensor refOut = iod["enc_out"];   // [1, 2T, 512]
        int t = (int)encIn.Shape[1];
        int dim = (int)encIn.Shape[2];

        using CpuBackend backend = new();
        using Tensor outT = enc.Forward(backend, encIn, dim);

        Assert.Equal(refOut.Shape[1], outT.Shape[1]);
        Assert.Equal(refOut.Shape[2], outT.Shape[2]);

        float* a = (float*)outT.DataPointer;
        float* b = (float*)refOut.DataPointer;
        long n = outT.ElementCount;
        double maxAbs = 0, sumSq = 0, refMax = 0;
        for (long i = 0; i < n; i++)
        {
            double diff = Math.Abs(a[i] - b[i]);
            if (diff > maxAbs) maxAbs = diff;
            sumSq += diff * diff;
            refMax = Math.Max(refMax, Math.Abs(b[i]));
        }
        double rms = Math.Sqrt(sumSq / n);
        _out.WriteLine($"T={t} → out [{outT.Shape[1]},{outT.Shape[2]}]. maxAbs={maxAbs:E4} rms={rms:E4} (ref peak {refMax:F4}).");
        _out.WriteLine($"  mine[0,0,:6]=[{a[0]:F5},{a[1]:F5},{a[2]:F5},{a[3]:F5},{a[4]:F5},{a[5]:F5}]");
        _out.WriteLine($"  ref [0,0,:6]=[{b[0]:F5},{b[1]:F5},{b[2]:F5},{b[3]:F5},{b[4]:F5},{b[5]:F5}]");
        Assert.True(maxAbs < 2e-3, $"encoder output diverges from reference (maxAbs={maxAbs:E4}).");
    }
}
