using HartsyInference.LLM.Multimodal;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>The reasoning markers are control tokens in the GGUF templates, and <c>Decode</c> drops control ids (as <c>GgufTokenizer</c> does). The
/// generator must still write the closing marker, or the reasoning split never finds it.</summary>
public sealed class Qwen35VlDecodeTests
{
    private const int OpenId = 1001, CloseId = 1002;

    [Fact]
    public void TheClosingMarkerSurvivesAsItsLiteralTextEvenWhenDecodeDropsControlIds()
    {
        ControlDroppingTokenizer tok = new();
        // "The bus." then </think> (a control id) then "A bus." — as the model generates it.
        int[] generated = [.. "The bus.".Select(c => (int)c), CloseId, .. "A bus.".Select(c => (int)c)];

        string literal = string.Concat(generated.Select(id => Qwen35VlGenerator.PieceText(tok, id, OpenId, CloseId)));
        Assert.Equal("The bus.</think>A bus.", literal);

        // The plain decode is the regression: it loses the marker.
        Assert.DoesNotContain("</think>", tok.Decode(generated));
    }

    [Fact]
    public void AnOrdinaryTokenIsDecodedAsBefore()
    {
        Assert.Equal("x", Qwen35VlGenerator.PieceText(new ControlDroppingTokenizer(), 'x', OpenId, CloseId));
    }

    /// <summary>Decode keeps ordinary ids and drops control ids, as GgufTokenizer does for its special set.</summary>
    private sealed class ControlDroppingTokenizer : ILlmTokenizer
    {
        public int[] Encode(string text, bool addSpecial) => EncodeOrdinary(text);
        public int[] EncodeOrdinary(string text) => text.Select(c => (int)c).ToArray();
        public string Decode(IReadOnlyList<int> ids) => string.Concat(ids.Where(i => i < 1000).Select(i => (char)i));
        public int? SpecialId(string token) => token switch { "<think>" => OpenId, "</think>" => CloseId, _ => null };
        public int? BosId => null;
        public int? EosId => null;
        public IReadOnlyList<int> StopIds => Array.Empty<int>();
        public string? BosToken => null;
        public string? EosToken => null;
    }
}
