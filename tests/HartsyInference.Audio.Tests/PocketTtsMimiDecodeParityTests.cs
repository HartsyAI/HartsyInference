using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.PocketTts;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Parity for the Pocket-TTS continuous-latent Mimi DECODE (latent[32] -> 24 kHz audio) against the
/// kyutai-labs/pocket-tts reference (ungated without-voice-cloning weights). Verifies output_proj + depthwise
/// upsample + 2-layer ProjectedTransformer (interleaved RoPE, sliding-window, LayerScale, GELU) + SEANet decoder.
///
/// <para>Gated on <c>POCKETTTS_WEIGHTS</c> (full model safetensors) + <c>POCKETTTS_REF_IO</c>
/// (<c>latent</c>/<c>audio_out</c>). Oracle: <c>tests/python-reference/pockettts_reference/</c>.</para></summary>
public sealed unsafe class PocketTtsMimiDecodeParityTests
{
    private readonly ITestOutputHelper _out;
    public PocketTtsMimiDecodeParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void MimiDecode_MatchesReference()
    {
        string? wPath = Environment.GetEnvironmentVariable("POCKETTTS_WEIGHTS");
        string? refIo = Environment.GetEnvironmentVariable("POCKETTTS_REF_IO");
        if (string.IsNullOrEmpty(wPath) || !File.Exists(wPath) || string.IsNullOrEmpty(refIo) || !File.Exists(refIo))
            return; // gated

        SafeTensorsLoader wl = new();
        wl.Load(wPath);
        IReadOnlyDictionary<string, Tensor> w = wl.GetAllTensors();

        PocketTtsMimiDecoder dec = new("mimi");
        dec.LoadWeights(w);

        SafeTensorsLoader io = new();
        io.Load(refIo);
        IReadOnlyDictionary<string, Tensor> d = io.GetAllTensors();
        Tensor latentRef = d["latent"];        // [1,32,T]
        Tensor audioRef = d["audio_out"];      // [1,1,S]
        int t = (int)latentRef.Shape[2];
        _out.WriteLine($"latent {latentRef.Shape}  audioRef {audioRef.Shape}  T={t}");

        using CpuBackend backend = new();
        Tensor pcm = dec.Forward(backend, latentRef, batch: 1, tFrames: t);
        _out.WriteLine($"my pcm {pcm.Shape}");
        Assert.Equal((int)audioRef.ElementCount, (int)pcm.ElementCount);

        (double corr, double rms, double maxAbs) = CorrRmsMax(pcm, audioRef);
        _out.WriteLine($"PocketTTS Mimi decode corr={corr:F6}  rms(mine)={rms:E4}  maxAbs={maxAbs:E4}");
        Assert.True(corr > 0.999, $"decode corr too low ({corr:F6}).");
        pcm.Dispose();
    }

    private static (double Corr, double Rms, double MaxAbs) CorrRmsMax(Tensor a, Tensor b)
    {
        float* pa = (float*)a.DataPointer;
        float* pb = (float*)b.DataPointer;
        long n = Math.Min(a.ElementCount, b.ElementCount);
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, sq = 0, mx = 0;
        for (long i = 0; i < n; i++)
        {
            double x = pa[i], y = pb[i];
            sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y; sq += x * x;
            mx = Math.Max(mx, Math.Abs(x - y));
        }
        double cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
        return (cov / (Math.Sqrt(va * vb) + 1e-12), Math.Sqrt(sq / n), mx);
    }
}
