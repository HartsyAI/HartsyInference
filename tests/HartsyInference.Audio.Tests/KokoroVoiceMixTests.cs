using System.IO;
using HartsyInference.Audio.Models.Kokoro;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Kokoro voice blends: the reference <c>KPipeline.load_voice("a,b")</c> averages the packs row by row; a
/// part may carry a weight.</summary>
public sealed class KokoroVoiceMixTests
{
    [Theory]
    [InlineData("af_heart", new[] { "af_heart" }, new[] { 1f })]
    [InlineData(" af_bella:0.7 , bm_lewis:0.3 ", new[] { "af_bella", "bm_lewis" }, new[] { 0.7f, 0.3f })]
    public void Parse_ReadsVoicesAndWeights(string spec, string[] voices, float[] weights)
    {
        KokoroVoiceMix mix = KokoroVoiceMix.Parse(spec);
        Assert.Equal(voices, mix.Parts.Select(p => p.Voice));
        Assert.Equal(weights, mix.Parts.Select(p => p.Weight));
        Assert.Equal(voices[0], mix.PrimaryVoice);
        Assert.Equal(voices.Length > 1, mix.IsBlend);
    }

    [Theory]
    [InlineData("../secrets")]
    [InlineData("af bella")]
    [InlineData("af_bella(1e3)")]
    public void Parse_RejectsWhatIsNotAVoice(string spec) => Assert.Throws<ArgumentException>(() => KokoroVoiceMix.Parse(spec));

    [Fact]
    public void Blend_IsTheWeightedMeanOfEachRow_OverTheSharedRows()
    {
        string dir = Directory.CreateTempSubdirectory("kokoro-mix").FullName;
        try
        {
            using KokoroVoicePack a = Pack(dir, "a", rows: 3, value: i => i);
            using KokoroVoicePack b = Pack(dir, "b", rows: 2, value: i => 10f * i);
            using KokoroVoicePack mean = KokoroVoiceMix.Parse("a,b").Blend("a,b", [a, b]);
            using KokoroVoicePack weighted = KokoroVoiceMix.Parse("a(3),b(1)").Blend("a(3),b(1)", [a, b]);
            Assert.Equal(2, mean.NumBuckets);
            using Tensor row = mean.GetStyle(2);
            using Tensor wrow = weighted.GetStyle(2);
            unsafe
            {
                float* r = (float*)row.DataPointer;
                float* w = (float*)wrow.DataPointer;
                int i = KokoroVoicePack.StyleWidth + 5; // row 1, column 5
                Assert.Equal((i + 10f * i) / 2f, r[5], 3);
                Assert.Equal((3f * i + 10f * i) / 4f, w[5], 3);
            }
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    private static KokoroVoicePack Pack(string dir, string name, int rows, Func<int, float> value)
    {
        string path = Path.Combine(dir, name + ".bin");
        using (BinaryWriter w = new(File.Create(path)))
        {
            for (int i = 0; i < rows * KokoroVoicePack.StyleWidth; i++) w.Write(value(i));
        }
        return KokoroVoicePack.LoadFromFile(path);
    }
}
