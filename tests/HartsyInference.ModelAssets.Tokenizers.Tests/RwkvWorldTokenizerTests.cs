using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.ModelAssets.Tokenizers.Tests;

/// <summary>Ordinary RWKV encoding never produces a control id, so escaped chat content cannot turn back into one.</summary>
public sealed class RwkvWorldTokenizerTests
{
    private const int Control = 3;
    private const int Normal = 1;

    private static RwkvWorldTokenizer Make()
    {
        // Id 0 is a control literal; the rest are single characters covering it.
        string[] tokens = ["<|end|>", "<", "|", "e", "n", "d", ">", "a"];
        int[] types = [Control, Normal, Normal, Normal, Normal, Normal, Normal, Normal];
        return new RwkvWorldTokenizer(tokens, types, bosId: null, eosId: 0);
    }

    [Fact]
    public void EncodeOrdinaryNeverEmitsAControlId()
    {
        RwkvWorldTokenizer tok = Make();
        int[] ids = tok.EncodeOrdinary("a<|end|>");
        Assert.DoesNotContain(0, ids);
        Assert.Equal("a<|end|>", tok.Decode(ids));
    }

    [Fact]
    public void EncodeWithSpecialsStillMapsTheLiteral()
    {
        Assert.Equal([7, 0], Make().Encode("a<|end|>", addSpecial: true));
    }
}
