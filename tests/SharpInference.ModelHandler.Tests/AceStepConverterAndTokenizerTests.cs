using Xunit;
using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.Tokenizers;

namespace SharpInference.ModelHandler.Tests;

/// <summary>ACE-Step checkpoint handling (weight-norm fusion math) and the VoiceBpe lyric tokenizer protocol
/// (start/separator tokens, [SPACE] handling, structure tags, language heuristic).</summary>
public unsafe class AceStepConverterAndTokenizerTests
{
    [Fact]
    public void FuseWeightNorm_MatchesPyTorchFormula()
    {
        // v rows: [3,4] (norm 5) and [0,2] (norm 2); g = [10, 6] ⇒ fused rows scaled to norm g.
        Tensor g = Make([10f, 6f], 2, 1);
        Tensor v = Make([3f, 4f, 0f, 2f], 2, 2);
        Dictionary<string, Tensor> raw = new()
        {
            ["conv.weight_g"] = g,
            ["conv.weight_v"] = v,
            ["conv.bias"] = Make([1f], 1, 1),
        };
        Dictionary<string, Tensor> fused = AceStepCheckpointConverter.FuseWeightNorm(raw);

        Assert.True(fused.ContainsKey("conv.weight"));
        Assert.True(fused.ContainsKey("conv.bias"));
        Assert.False(fused.ContainsKey("conv.weight_g"));
        Assert.False(fused.ContainsKey("conv.weight_v"));
        float* p = (float*)fused["conv.weight"].DataPointer;
        Assert.Equal(6f, p[0], 4);     // 3/5 · 10
        Assert.Equal(8f, p[1], 4);     // 4/5 · 10
        Assert.Equal(0f, p[2], 4);
        Assert.Equal(6f, p[3], 4);     // 2/2 · 6
    }

    [Fact]
    public void LyricTokenizer_FollowsTheAceStepLineProtocol()
    {
        (string vocabPath, string mergesPath) = WriteTinyTokenizer();
        try
        {
            AceStepLyricTokenizer tokenizer = new(vocabPath, mergesPath);

            // "[261] | [en] h i [SPACE] h i | [2] | [2] (blank) | [en]([verse] forced en) v… | [2]"
            int[] ids = tokenizer.TokenizeLyrics("hi hi\n\n[verse]");
            Assert.Equal(AceStepLyricTokenizer.StartToken, ids[0]);
            Assert.Equal(AceStepLyricTokenizer.LineSeparatorToken, ids[^1]);
            Assert.Equal(3, ids.Count(t => t == AceStepLyricTokenizer.LineSeparatorToken));
            Assert.Contains(30, ids);                       // [en] prefix token
            Assert.Contains(31, ids);                       // [SPACE]
            Assert.Contains(20, ids);                       // merged "hi"

            // Explicit-language override flows through; han text auto-detects zh.
            Assert.Equal("zh", AceStepLyricTokenizer.DetectLanguage("你好"));
            Assert.Equal("ko", AceStepLyricTokenizer.DetectLanguage("안녕"));
            Assert.Equal("en", AceStepLyricTokenizer.DetectLanguage("hello"));
        }
        finally
        {
            File.Delete(vocabPath);
            File.Delete(mergesPath);
        }
    }

    private static (string Vocab, string Merges) WriteTinyTokenizer()
    {
        string vocab = Path.Combine(Path.GetTempPath(), $"ace_vocab_{Guid.NewGuid():N}.json");
        string merges = Path.Combine(Path.GetTempPath(), $"ace_merges_{Guid.NewGuid():N}.txt");
        File.WriteAllText(vocab, """{"h": 10, "i": 11, "v": 12, "e": 13, "r": 14, "s": 15, "[": 16, "]": 17, "hi": 20, "[en]": 30, "[SPACE]": 31}""");
        File.WriteAllText(merges, "h i\n");
        return (vocab, merges);
    }

    private static Tensor Make(float[] values, int rows, int cols)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        values.CopyTo(new Span<float>((float*)t.DataPointer, values.Length));
        return t;
    }
}
