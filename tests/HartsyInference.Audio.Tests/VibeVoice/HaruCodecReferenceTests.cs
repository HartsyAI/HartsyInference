using System.Text.RegularExpressions;
using HartsyInference.Audio.Models.VibeVoice;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.VibeVoice;

/// <summary>The engine's VibeVoice codec classes, fed Haru's codec tensors through a key map, against the upstream Haru codec
/// (<c>tools/haru/haru_reference.py</c>, CPU F32) on a tiny random checkpoint: a whole clip ending mid-frame, a whole decode,
/// and frame-by-frame streaming in both directions.</summary>
public sealed unsafe class HaruCodecReferenceTests(ITestOutputHelper output)
{
    private const float RelativeTolerance = 2e-5f;
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Haru");

    private static string RemapKey(string key)
    {
        string k = key["codec.".Length..];
        k = Regex.Replace(k, @"^encoder\.stem\.", "encoder.downsample_layers.0.0.conv.conv.");
        k = Regex.Replace(k, @"^encoder\.down\.(\d+)\.", m => $"encoder.downsample_layers.{int.Parse(m.Groups[1].Value) + 1}.0.conv.conv.");
        k = Regex.Replace(k, @"^decoder\.stem\.", "decoder.upsample_layers.0.0.conv.conv.");
        k = Regex.Replace(k, @"^decoder\.up\.(\d+)\.", m => $"decoder.upsample_layers.{int.Parse(m.Groups[1].Value) + 1}.0.convtr.convtr.");
        k = Regex.Replace(k, @"\.stages\.(\d+)\.(\d+)\.conv\.", ".stages.$1.$2.mixer.conv.conv.conv.");
        k = Regex.Replace(k, @"^(encoder|decoder)\.head\.", "$1.head.conv.conv.");
        k = k.Replace(".fc1.", ".ffn.linear1.").Replace(".fc2.", ".ffn.linear2.");
        return "tok." + k;
    }

    private static float[] Read(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    private static Tensor Make(float[] values, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        values.AsSpan().CopyTo(new Span<float>((void*)t.DataPointer, values.Length));
        return t;
    }

    private void Check(string name, float[] got, float[] want)
    {
        Assert.Equal(want.Length, got.Length);
        float worst = 0;
        float scale = 1e-12f;
        for (int i = 0; i < got.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(got[i] - want[i]));
            scale = MathF.Max(scale, MathF.Abs(want[i]));
        }
        output.WriteLine($"{name}: n={got.Length} max|d|={worst:E3} max|ref|={scale:E3} rel={worst / scale:E3}");
        Assert.True(worst / scale < RelativeTolerance, $"{name}: relative error {worst / scale:E3}");
    }

    [Fact]
    public void Codec_MatchesUpstreamHaru()
    {
        using SafeTensorsLoader weights = new();
        weights.Load(Path.Combine(Dir, "haru_tiny.safetensors"));
        using SafeTensorsLoader expected = new();
        expected.Load(Path.Combine(Dir, "haru_tiny_expected.safetensors"));
        Dictionary<string, Tensor> codec = new(StringComparer.Ordinal);
        foreach (string name in weights.Descriptors.Keys.Where(static n => n.StartsWith("codec.", StringComparison.Ordinal)))
        {
            codec[RemapKey(name)] = weights.GetTensor(name);
        }
        VibeVoiceTokenizerConfig config = new()
        {
            EncoderNFilters = 1, DecoderNFilters = 1, VaeDim = 16,
            EncoderRatios = [8, 5, 5, 4, 2, 2], DecoderRatios = [8, 5, 5, 4, 2, 2],
            EncoderDepths = "1-1-1-1-1-1-2", DecoderDepths = "2-1-1-1-1-1-1",
        };
        VibeVoiceAcousticTokenizerModel model = new(config, "tok");
        model.LoadWeights(codec);
        using CpuBackend backend = new();

        float[] clip = Read(expected.GetTensor("in.clip"));
        using Tensor pcm = Make(clip, 1, 1, clip.Length);
        using Tensor mean = model.Encode(backend, pcm, 1, clip.Length);
        Assert.Equal(6, (int)mean.Shape[1]);
        Check("whole clip encode (ends mid-frame)", Read(mean), Read(expected.GetTensor("ref.clip_mean")));

        float[] latents = Read(expected.GetTensor("in.latents"));
        using Tensor lat = Make(latents, 1, 3, 16);
        using Tensor audio = model.Decode(backend, lat, 1);
        Check("whole decode", Read(audio), Read(expected.GetTensor("ref.clip_audio")));

        float[] frames = Read(expected.GetTensor("in.frames"));
        using VibeVoiceTokenizerStreamingCache encoderCache = new();
        List<float> streamedMean = [];
        for (int f = 0; f < 3; f++)
        {
            using Tensor one = Make(frames[(f * 3200)..((f + 1) * 3200)], 1, 1, 3200);
            using Tensor m = model.Encode(backend, one, 1, 3200, encoderCache, [0]);
            streamedMean.AddRange(Read(m));
        }
        Check("streamed encode", [.. streamedMean], Read(expected.GetTensor("ref.stream_mean")));

        using VibeVoiceTokenizerStreamingCache decoderCache = new();
        List<float> streamedAudio = [];
        for (int f = 0; f < 3; f++)
        {
            using Tensor one = Make(latents[(f * 16)..((f + 1) * 16)], 1, 1, 16);
            using Tensor a = model.Decode(backend, one, 1, decoderCache, [0]);
            streamedAudio.AddRange(Read(a));
        }
        Check("streamed decode", [.. streamedAudio], Read(expected.GetTensor("ref.stream_audio")));
    }
}
