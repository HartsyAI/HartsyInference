using System;
using System.Collections.Generic;
using HartsyInference.Audio.Models.QwenTts;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weights verification for the Qwen3-TTS codec decoder. Loads the actual
/// <c>Qwen/Qwen3-TTS-12Hz-1.7B-Base/speech_tokenizer/model.safetensors</c> (the <c>decoder.*</c> sub-tree) and
/// confirms (1) every key the vocoder reads exists in the checkpoint and (2) a decode runs end to end to finite
/// 24 kHz PCM of length <c>T * 1920</c>.
///
/// <para>Gated on <c>QWEN3TTS_CODEC_PATH</c> pointing at the downloaded safetensors (the file is ~682 MB, so it
/// is not committed). Skips cleanly when unset.</para></summary>
public sealed class Qwen3TtsCodecRealWeightsTests
{
    private readonly ITestOutputHelper _out;
    public Qwen3TtsCodecRealWeightsTests(ITestOutputHelper o) => _out = o;

    private static string? CodecPath => Environment.GetEnvironmentVariable("QWEN3TTS_CODEC_PATH");

    [Fact]
    [Trait("Category", "Integration")]
    public void LoadsRealDecoderKeys_AndDecodesToFinitePcm()
    {
        string? path = CodecPath;
        if (string.IsNullOrEmpty(path) || !System.IO.File.Exists(path))
        {
            return; // gated: no checkpoint present
        }

        SafeTensorsLoader loader = new();
        loader.Load(path);
        IReadOnlyDictionary<string, Tensor> w = loader.GetAllTensors();

        using Qwen3TtsVocoder vocoder = new(Qwen3TtsVocoderConfig.Default);
        vocoder.LoadWeights(w, "decoder");   // throws if any decoder.* key it needs is missing

        // Synthetic 16-codebook grid (row 0 = semantic, 1..15 = acoustic), valid code range [0, 2048).
        const int t = 8;
        int[,] grid = new int[16, t];
        uint rng = 0x1234_5678u;
        for (int r = 0; r < 16; r++)
            for (int s = 0; s < t; s++)
            {
                rng = rng * 1664525u + 1013904223u;
                grid[r, s] = (int)(rng % 2048u);
            }

        using CpuBackend backend = new();
        float[] pcm = vocoder.Decode(backend, grid);

        Assert.Equal(t * vocoder.Config.TotalUpsample, pcm.Length);   // 8 * 1920 = 15360
        double sumSq = 0; float peak = 0;
        foreach (float v in pcm)
        {
            Assert.True(float.IsFinite(v), "decoded PCM must be finite");
            sumSq += (double)v * v; peak = Math.Max(peak, Math.Abs(v));
        }
        double rms = Math.Sqrt(sumSq / pcm.Length);
        _out.WriteLine($"Codec decode of random codes: RMS={rms:0.0000} peak={peak:0.0000} (saturated RMS~1.0 => decoder numerical bug; bounded RMS<0.5 => decoder OK, bug is upstream).");
    }
}
