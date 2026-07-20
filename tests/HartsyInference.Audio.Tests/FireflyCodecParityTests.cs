using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.FishSpeech;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight parity for the Fish-Speech 1.5 firefly-gan-vq codec DECODE (indices -> 44.1 kHz audio)
/// vs the upstream FireflyArchitecture (fish-speech v1.5.1). Exercises the grouped-residual FSQ dequant +
/// ConvNeXt upsample + SiLU HiFi-GAN. Gated on <c>FIREFLY_WEIGHTS</c> + <c>FIREFLY_REF</c>
/// (dump_firefly_reference.py).</summary>
public sealed unsafe class FireflyCodecParityTests
{
    private readonly ITestOutputHelper _out;
    public FireflyCodecParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void FireflyDecode_MatchesReference()
    {
        string? wPath = Environment.GetEnvironmentVariable("FIREFLY_WEIGHTS");
        string? refP = Environment.GetEnvironmentVariable("FIREFLY_REF");
        if (string.IsNullOrEmpty(wPath) || !File.Exists(wPath) || string.IsNullOrEmpty(refP) || !File.Exists(refP))
            return;

        SafeTensorsLoader wl = new(); wl.Load(wPath);
        IReadOnlyDictionary<string, Tensor> w = wl.GetAllTensors();
        SafeTensorsLoader rl = new(); rl.Load(refP);
        IReadOnlyDictionary<string, Tensor> r = rl.GetAllTensors();

        Tensor idxRef = r["indices"];        // [1,8,T] int32
        Tensor audioRef = r["audio_out"];    // [1,1,S]
        int groups = (int)idxRef.Shape[1];
        int t = (int)idxRef.Shape[2];
        int[,] codes = new int[groups, t];
        int* ip = (int*)idxRef.DataPointer;
        for (int g = 0; g < groups; g++)
            for (int j = 0; j < t; j++) codes[g, j] = ip[(long)g * t + j];

        using FireflyDecoder dec = new(numCodebooks: groups, codebookSize: 8 * 5 * 5 * 5, config: FireflyConfig.V1_5);
        dec.LoadWeights(w, "head", "quantizer");

        using CpuBackend backend = new();
        float[] audio = dec.Decode(backend, codes, t);
        _out.WriteLine($"my audio len {audio.Length}  ref {audioRef.ElementCount}");
        Assert.Equal((int)audioRef.ElementCount, audio.Length);

        float* br = (float*)audioRef.DataPointer;
        long n = audio.Length;
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, mx = 0;
        for (long i = 0; i < n; i++)
        {
            double x = audio[i], y = br[i];
            sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
            mx = Math.Max(mx, Math.Abs(x - y));
        }
        double cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
        double corr = cov / (Math.Sqrt(va * vb) + 1e-12);
        _out.WriteLine($"Firefly decode corr={corr:F6}  maxAbs={mx:E4}");
        Assert.True(corr > 0.999, $"firefly decode corr too low ({corr:F6}).");
    }
}
