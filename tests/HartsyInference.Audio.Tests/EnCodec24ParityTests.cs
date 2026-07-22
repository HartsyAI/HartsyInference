using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Codecs.EnCodec;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;
using EnCodecModel = HartsyInference.Audio.Models.Codecs.EnCodec.EnCodec;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight parity for EnCodec 24 kHz DECODE (Bark's codec, causal) vs `transformers` EncodecModel
/// (facebook/encodec_24khz). Complements the validated 32 kHz path (MusicGen). Gated on
/// <c>ENCODEC24_WEIGHTS</c> + <c>ENCODEC24_REF</c> (dump_encodec24_reference.py).</summary>
public sealed unsafe class EnCodec24ParityTests
{
    private readonly ITestOutputHelper _out;
    public EnCodec24ParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void EnCodec24Decode_MatchesReference()
    {
        string? wPath = Environment.GetEnvironmentVariable("ENCODEC24_WEIGHTS");
        string? refP = Environment.GetEnvironmentVariable("ENCODEC24_REF");
        if (string.IsNullOrEmpty(wPath) || !File.Exists(wPath) || string.IsNullOrEmpty(refP) || !File.Exists(refP))
            return;

        SafeTensorsLoader wl = new(); wl.Load(wPath);
        Dictionary<string, Tensor> w = MusicGenCheckpointConverter.MapEnCodecWeights(wl.GetAllTensors(), castToF32: true);
        SafeTensorsLoader rl = new(); rl.Load(refP);
        IReadOnlyDictionary<string, Tensor> r = rl.GetAllTensors();

        Tensor codesRef = r["codes"];        // [nQ, B, T] int32
        Tensor audioRef = r["audio_out"];    // [1,1,S]
        int nq = (int)codesRef.Shape[0];
        int batch = (int)codesRef.Shape[1];
        int t = (int)codesRef.Shape[2];

        EnCodecModel codec = new(EnCodecConfig.EnCodec24kHz);
        codec.LoadWeights(w);

        Tensor codes = new(new TensorShape(nq, batch, t), DType.I32);
        int* cp = (int*)codes.DataPointer; int* cr = (int*)codesRef.DataPointer;
        for (long i = 0; i < codes.ElementCount; i++) cp[i] = cr[i];

        using CpuBackend backend = new();
        Tensor pcm = codec.Decode(backend, codes, batch, t);
        _out.WriteLine($"my pcm {pcm.Shape}  ref {audioRef.Shape}");
        Assert.Equal((int)audioRef.ElementCount, (int)pcm.ElementCount);

        float* a = (float*)pcm.DataPointer; float* b = (float*)audioRef.DataPointer;
        long n = pcm.ElementCount;
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, mx = 0;
        for (long i = 0; i < n; i++)
        {
            double x = a[i], y = b[i];
            sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
            mx = Math.Max(mx, Math.Abs(x - y));
        }
        double cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
        double corr = cov / (Math.Sqrt(va * vb) + 1e-12);
        _out.WriteLine($"EnCodec24 decode corr={corr:F6}  maxAbs={mx:E4}");
        Assert.True(corr > 0.999, $"EnCodec24 decode corr too low ({corr:F6}).");
        codes.Dispose(); pcm.Dispose();
    }
}
