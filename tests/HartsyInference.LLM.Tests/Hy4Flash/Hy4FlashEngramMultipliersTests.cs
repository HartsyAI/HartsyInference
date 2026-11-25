using HartsyInference.LLM.Hy4Flash;

namespace HartsyInference.LLM.Tests.Hy4Flash;

/// <summary>Expected values come from numpy 2.5.4: <c>default_rng(seed + 10007 * layer_id).integers(0, int64_max // vocab // 2, size=3, dtype=np.int64) * 2 + 1</c>.</summary>
public sealed class Hy4FlashEngramMultipliersTests
{
    [Theory]
    [InlineData(2, 120818, new long[] { 62749575316035, 30426722171175, 18574837598557 })]
    [InlineData(12, 120818, new long[] { 14916611332703, 34290019069965, 64992498366675 })]
    [InlineData(22, 120818, new long[] { 55942601167937, 16505523750331, 42928537650479 })]
    [InlineData(32, 120818, new long[] { 22330137526737, 12836963598171, 17229190754683 })]
    [InlineData(2, 120000, new long[] { 63177318254439, 30634130993975, 18701456074853 })]
    [InlineData(32, 250, new long[] { 10791530222821615, 6203745072015221, 8326385474396959 })]
    public void Draw_MatchesNumpyForTheRealLayerIdsAndVocabularySizes(int layerId, int vocab, long[] expected) =>
        Assert.Equal(expected, Hy4FlashEngramMultipliers.Draw(0, layerId, vocab, 3));

    [Fact]
    public void Draw_UsesTheConfigLayerIdNotTheDecoderIndex() =>
        Assert.NotEqual(Hy4FlashEngramMultipliers.Draw(0, 2, 120818, 3), Hy4FlashEngramMultipliers.Draw(0, 1, 120818, 3));
}
