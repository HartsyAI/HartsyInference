using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Codecs.NeuCodec;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Dumps OUR <see cref="NeuCodecEncoder"/> FSQ code indices for the fixed 16 kHz jfk clip so they can
/// be diffed against the official <c>neucodec</c> reference (tests/python-reference/neucodec_ref.py →
/// /tmp/neucodec_ref_codes.npy). Self-discovers the on-disk <c>neuphonic--neucodec/model.safetensors</c>
/// (transformers-format keys: <c>acoustic_encoder.* / semantic_encoder.* / semantic_adapter.* / fc_encoder /
/// quantizer.*</c>) from the model cache and skips cleanly when weights or clip are absent.
/// <para>Writes <c>/tmp/neucodec_ours_codes.txt</c> (whitespace-separated ints) plus shape/min/max/first-32.
/// Runs on CPU; the coordinator runs it (the encoder needs a backend).</para></summary>
public sealed class NeuCodecEncoderDumpTest
{
    private readonly ITestOutputHelper _out;
    public NeuCodecEncoderDumpTest(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void DumpOurNeuCodecCodes()
    {
        string weights = Path.Combine(AudioModelCache.CacheRoot, "neuphonic--neucodec", "model.safetensors");
        string jfk = Path.Combine(AudioModelCache.CacheRoot, "test-clips", "jfk.wav");
        if (!File.Exists(weights)) { _out.WriteLine($"skip: no weights at {weights}"); return; }
        if (!File.Exists(jfk)) { _out.WriteLine($"skip: no clip at {jfk}"); return; }

        // Fixed clip → mono 16 kHz float PCM (jfk is already 16 kHz mono).
        WavFile.DecodedAudio a = WavFile.Read(jfk);
        float[] pcm = a.Channels[0];
        Assert.Equal(16_000, a.SampleRate);

        using SafeTensorsLoader loader = new();
        loader.Load(weights);
        Dictionary<string, Tensor> w = loader.GetAllTensors();

        using CpuBackend backend = new();
        using NeuCodecEncoder enc = new(NeuCodecEncoderConfig.Default);
        enc.LoadWeights(w);

        int[] codes = enc.Encode(backend, pcm);

        File.WriteAllText("/tmp/neucodec_ours_codes.txt", string.Join(" ", codes));
        int min = codes.Length == 0 ? 0 : codes.Min();
        int max = codes.Length == 0 ? 0 : codes.Max();
        _out.WriteLine($"codes shape: [{codes.Length}]");
        _out.WriteLine($"codes min/max: {min} / {max}");
        _out.WriteLine($"first 32 codes: {string.Join(", ", codes.Take(32))}");

        Assert.NotEmpty(codes);
        Assert.All(codes, c => Assert.InRange(c, 0, 65_535));
    }
}
