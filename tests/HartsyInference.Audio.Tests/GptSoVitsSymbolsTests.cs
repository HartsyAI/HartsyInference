using HartsyInference.Audio.Frontends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Verifies the GPT-SoVITS v2 symbol table and BERT word2ph expansion against the upstream
/// <c>cleaned_text_to_sequence</c> mapping (checkpoint-free).</summary>
public sealed unsafe class GptSoVitsSymbolsTests
{
    [Fact]
    public void ToSequence_MatchesUpstreamCleanedTextToSequence()
    {
        Assert.Equal(732, GptSoVitsSymbols.Count);
        string[] syms = ["!", "AA", "zh", "ang1", "SP", "UNK", "."];
        int[] expected = [0, 5, 320, 112, 77, 86, 3];     // upstream cleaned_text_to_sequence(version='v2')
        Assert.Equal(expected, GptSoVitsSymbols.ToSequence(syms));
    }

    [Fact]
    public void ExpandBertToPhonemes_RepeatsPerWord2ph()
    {
        // 3 chars, dim 2, word2ph [2,1,3] → 6 phoneme columns.
        Tensor charBert = new(new TensorShape(2, 3), DType.F32);
        float* p = (float*)charBert.DataPointer;
        p[0] = 1; p[1] = 2; p[2] = 3;       // dim 0
        p[3] = 4; p[4] = 5; p[5] = 6;       // dim 1
        int[] word2ph = [2, 1, 3];

        Tensor outT = GptSoVitsSymbols.ExpandBertToPhonemes(charBert, word2ph);
        try
        {
            Assert.Equal(6, (int)outT.Shape[1]);
            float* o = (float*)outT.DataPointer;
            Assert.Equal(new float[] { 1, 1, 2, 3, 3, 3 }, new[] { o[0], o[1], o[2], o[3], o[4], o[5] });
            Assert.Equal(new float[] { 4, 4, 5, 6, 6, 6 }, new[] { o[6], o[7], o[8], o[9], o[10], o[11] });
        }
        finally { outT.Dispose(); charBert.Dispose(); }
    }
}
