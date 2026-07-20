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

/// <summary>Numerical parity for the <see cref="HiFTNetVocoder"/> against the real <c>s3gen.safetensors</c>
/// <c>mel2wav.*</c>. Feeds the fixed PyTorch-reference mel and diffs the 24 kHz waveform
/// (<c>ref_vocoder_io.safetensors</c>: mel [1,80,T] → wav [1,T·480]). Gated on <c>S3GEN_PATH</c> +
/// <c>REF_VOCODER_IO</c>.</summary>
public sealed unsafe class CosyVoiceVocoderParityTests
{
    private readonly ITestOutputHelper _out;
    public CosyVoiceVocoderParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void Vocoder_MatchesPythonReference()
    {
        string? s3gen = Environment.GetEnvironmentVariable("S3GEN_PATH");
        string? refIo = Environment.GetEnvironmentVariable("REF_VOCODER_IO");
        if (string.IsNullOrEmpty(s3gen) || !File.Exists(s3gen) || string.IsNullOrEmpty(refIo) || !File.Exists(refIo))
            return; // gated

        SafeTensorsLoader sl = new();
        sl.Load(s3gen);
        Dictionary<string, Tensor> vw = new();
        const string pre = "mel2wav.";
        foreach (KeyValuePair<string, Tensor> kv in sl.GetAllTensors())
            if (kv.Key.StartsWith(pre, StringComparison.Ordinal)) vw[kv.Key[pre.Length..]] = kv.Value;
        _out.WriteLine($"mel2wav.* tensors: {vw.Count}");

        using HiFTNetVocoder voc = new(CosyVoiceConfig.V2_0_5B.Hift);
        voc.LoadWeights(vw);

        SafeTensorsLoader io = new();
        io.Load(refIo);
        IReadOnlyDictionary<string, Tensor> d = io.GetAllTensors();
        Tensor mel = d["mel"];
        Tensor refWav = d["wav"];
        int n = (int)refWav.ElementCount;

        using CpuBackend backend = new();
        float[] wav = voc.Forward(backend, mel);
        _out.WriteLine($"mine wav length {wav.Length}, ref {n}.");

        int cmp = Math.Min(wav.Length, n);
        float* rw = (float*)refWav.DataPointer;
        double maxAbs = 0, refPeak = 0;
        for (int i = 0; i < cmp; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(wav[i] - rw[i]));
            refPeak = Math.Max(refPeak, Math.Abs(rw[i]));
        }
        _out.WriteLine($"maxAbs={maxAbs:E4} (ref peak {refPeak:F4}). mine[:6]=[{wav[0]:F5},{wav[1]:F5},{wav[2]:F5},{wav[3]:F5},{wav[4]:F5},{wav[5]:F5}]");
        _out.WriteLine($"  ref[:6]=[{rw[0]:F5},{rw[1]:F5},{rw[2]:F5},{rw[3]:F5},{rw[4]:F5},{rw[5]:F5}]");
        Assert.Equal(n, wav.Length);
        Assert.True(maxAbs < 5e-3, $"vocoder output diverges (maxAbs={maxAbs:E4}).");
    }
}
