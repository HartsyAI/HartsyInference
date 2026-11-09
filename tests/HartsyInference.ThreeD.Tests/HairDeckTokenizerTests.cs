using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.HairDeck;
using HartsyInference.ThreeD.Tests.Parity;

namespace HartsyInference.ThreeD.Tests;

public sealed class HairDeckTokenizerTests
{
    private static HairDeckTokenizer Identity() =>
        new(new float[3], new float[] { 1f, 1f, 1f }, new float[HairDeckTokenizer.LatentDim], Enumerable.Repeat(1f, HairDeckTokenizer.LatentDim).ToArray());

    [Fact]
    public void Clean_ZeroesMissingSlots_AndSnapsExistingOnes()
    {
        float[] raw = new float[2 * HairDeckTokenizer.TokenDim];
        raw[35] = 3f; raw[36] = 0f; raw[37] = 4f;
        raw[38] = 0f;
        raw[39] = 0.5f;
        raw[HairDeckTokenizer.TokenDim + 3] = 7f;
        raw[HairDeckTokenizer.TokenDim + 39] = -0.1f;
        using Tensor tokens = HairDeckParityComponents.Make(raw, 2, HairDeckTokenizer.TokenDim);
        using Tensor clean = Identity().Clean(tokens);
        float[] y = HairDeckNumerics.ToArray(clean);

        Assert.Equal(0.6f, y[35], 5);
        Assert.Equal(0f, y[36], 5);
        Assert.Equal(0.8f, y[37], 5);
        Assert.Equal(1f, y[38]);
        Assert.Equal(1f, y[39]);
        for (int k = HairDeckTokenizer.TokenDim; k < 2 * HairDeckTokenizer.TokenDim - 1; k++) Assert.Equal(0f, y[k]);
        Assert.Equal(-1f, y[2 * HairDeckTokenizer.TokenDim - 1]);
    }

    [Fact]
    public void SignNormals_FlipsOnlyCardsPointingAgainstTheirRootOffset()
    {
        float[] raw = new float[2 * HairDeckTokenizer.TokenDim];
        raw[0] = 0f; raw[1] = 0f; raw[2] = 0f;
        raw[35] = 1f; raw[39] = 1f;
        raw[HairDeckTokenizer.TokenDim] = 2f;
        raw[HairDeckTokenizer.TokenDim + 35] = 1f;
        raw[HairDeckTokenizer.TokenDim + 39] = 1f;
        using Tensor tokens = HairDeckParityComponents.Make(raw, 2, HairDeckTokenizer.TokenDim);
        using Tensor signed = Identity().SignNormals(tokens);
        float[] y = HairDeckNumerics.ToArray(signed);

        Assert.Equal(-1f, y[35]);
        Assert.Equal(1f, y[HairDeckTokenizer.TokenDim + 35]);
    }

    [Fact]
    public void DecodeSlot_UndoesTheStandardisation()
    {
        float[] latentMu = Enumerable.Range(0, HairDeckTokenizer.LatentDim).Select(i => (float)i).ToArray();
        float[] latentSd = Enumerable.Repeat(2f, HairDeckTokenizer.LatentDim).ToArray();
        HairDeckTokenizer tokenizer = new([1f, 2f, 3f], [4f, 5f, 6f], latentMu, latentSd);
        float[] raw = new float[HairDeckTokenizer.TokenDim];
        raw[0] = 1f; raw[1] = 1f; raw[2] = 1f;
        raw[3] = 0.5f;
        raw[38] = 1f; raw[39] = 1f;
        using Tensor tokens = HairDeckParityComponents.Make(raw, 1, HairDeckTokenizer.TokenDim);
        Span<float> root = stackalloc float[3];
        Span<float> latent = stackalloc float[HairDeckTokenizer.LatentDim];
        Span<float> normal = stackalloc float[3];
        tokenizer.DecodeSlot(tokens, 0, root, latent, normal, out int shape, out bool exists);

        Assert.Equal(5f, root[0]);
        Assert.Equal(7f, root[1]);
        Assert.Equal(9f, root[2]);
        Assert.Equal(1f, latent[0]);
        Assert.Equal(1 + 0f, latent[0]);
        Assert.Equal(1, shape);
        Assert.True(exists);
    }
}
