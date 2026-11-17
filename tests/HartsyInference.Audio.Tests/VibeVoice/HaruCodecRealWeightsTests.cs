using System.Text.Json;
using HartsyInference.Audio.Models.Haru;
using HartsyInference.Audio.Models.VibeVoice;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.VibeVoice;

/// <summary>The released Haru-4B codec (<c>higurashi-ai/Haru-4B</c> @ 8d6e83bc, <c>codec.*</c> tensors only) against the upstream
/// PyTorch codec on a real recording, CPU F32. <c>HARU_CODEC_REAL_DIR</c> names a folder with <c>haru_codec_only.safetensors</c>
/// (the <c>codec.*</c> byte range of the checkpoint as a standalone file) and <c>codec_real_expected.safetensors</c> (the
/// upstream outputs on the clip).</summary>
[Trait("Category", "Integration")]
public sealed unsafe class HaruCodecRealWeightsTests(ITestOutputHelper output)
{
    private const float RelativeTolerance = 5e-5f;

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
    public void ReleasedCodec_MatchesUpstreamOnARealRecording()
    {
        string? dir = Environment.GetEnvironmentVariable("HARU_CODEC_REAL_DIR");
        string weightsPath = Path.Combine(dir ?? string.Empty, "haru_codec_only.safetensors");
        string expectedPath = Path.Combine(dir ?? string.Empty, "codec_real_expected.safetensors");
        if (string.IsNullOrEmpty(dir) || !File.Exists(weightsPath) || !File.Exists(expectedPath))
        {
            Assert.False(Environment.GetEnvironmentVariable("HARTSY_REQUIRE_REAL_WEIGHTS") == "1",
                "HARU_CODEC_REAL_DIR does not hold haru_codec_only.safetensors and codec_real_expected.safetensors.");
            return;
        }

        using SafeTensorsLoader weights = new();
        weights.Load(weightsPath);
        using SafeTensorsLoader expected = new();
        expected.Load(expectedPath);
        using JsonDocument config = JsonDocument.Parse(HaruCodecWeightsTests.ReleasedCodec);
        VibeVoiceAcousticTokenizerModel model = HaruCodec.Create(config.RootElement, weights.GetAllTensors());
        using CpuBackend backend = new();

        float[] clip = Read(expected.GetTensor("in.clip"));
        using Tensor pcm = Make(clip, 1, 1, clip.Length);
        using Tensor mean = model.Encode(backend, pcm, 1, clip.Length);
        float[] wantMean = Read(expected.GetTensor("ref.mean"));
        Check("whole clip encode", Read(mean), wantMean);

        int frames = (int)mean.Shape[1];
        using Tensor lat = Make(wantMean, 1, frames, 64);
        using Tensor audio = model.Decode(backend, lat, 1);
        Check("whole decode", Read(audio), Read(expected.GetTensor("ref.audio")));

        float[] firstFrames = Read(expected.GetTensor("in.frames"));
        using VibeVoiceTokenizerStreamingCache encoderCache = new();
        List<float> streamedMean = [];
        for (int f = 0; f < 3; f++)
        {
            using Tensor one = Make(firstFrames[(f * 3200)..((f + 1) * 3200)], 1, 1, 3200);
            using Tensor m = model.Encode(backend, one, 1, 3200, encoderCache, [0]);
            streamedMean.AddRange(Read(m));
        }
        Check("streamed encode", [.. streamedMean], Read(expected.GetTensor("ref.stream_mean")));

        using VibeVoiceTokenizerStreamingCache decoderCache = new();
        List<float> streamedAudio = [];
        for (int f = 0; f < 3; f++)
        {
            using Tensor one = Make(wantMean[(f * 64)..((f + 1) * 64)], 1, 1, 64);
            using Tensor a = model.Decode(backend, one, 1, decoderCache, [0]);
            streamedAudio.AddRange(Read(a));
        }
        Check("streamed decode", [.. streamedAudio], Read(expected.GetTensor("ref.stream_audio")));
    }
}
