using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.HairDeck;
using HartsyInference.ThreeD.Tests.Parity;

namespace HartsyInference.ThreeD.Tests;

public sealed class HairDeckChainDecoderTests
{
    [Fact]
    public void Decode_ResamplesThePathAtArcLengthPositions()
    {
        float[] raw = new float[2 * HairDeckChainDecoder.TokenDim];
        for (int i = 0; i < HairDeckChainDecoder.PathSamples; i++) raw[i * 3] = i;
        raw[24] = 4f;
        raw[25] = 0.5f;
        raw[26] = 1f;
        raw[HairDeckChainDecoder.TokenDim + 26] = -1f;
        using Tensor tokens = HairDeckParityComponents.Make(raw, 2, HairDeckChainDecoder.TokenDim);
        float[] mu = new float[26], sd = Enumerable.Repeat(1f, 26).ToArray();

        IReadOnlyList<DecodedChain> chains = HairDeckChainDecoder.Decode(tokens, mu, sd);

        Assert.Single(chains);
        DecodedChain chain = chains[0];
        Assert.Equal(4, chain.JointCount);
        Assert.Equal(0.5f, chain.Stiffness);
        Assert.Equal(12, chain.Joints.Length);
        float[] expectedX = [0f, 7f / 3f, 14f / 3f, 7f];
        for (int j = 0; j < 4; j++)
        {
            Assert.Equal(expectedX[j], chain.Joints[j * 3], 4);
            Assert.Equal(0f, chain.Joints[j * 3 + 1], 6);
        }
        Assert.Equal(24, chain.Path.Length);
    }

    [Fact]
    public void Decode_ClampsTheJointCountAndStiffness()
    {
        float[] raw = new float[HairDeckChainDecoder.TokenDim];
        raw[24] = 0.2f;
        raw[25] = 3f;
        raw[26] = 1f;
        using Tensor tokens = HairDeckParityComponents.Make(raw, 1, HairDeckChainDecoder.TokenDim);
        IReadOnlyList<DecodedChain> chains = HairDeckChainDecoder.Decode(tokens, new float[26], Enumerable.Repeat(1f, 26).ToArray());

        Assert.Equal(2, chains[0].JointCount);
        Assert.Equal(1f, chains[0].Stiffness);
    }
}
