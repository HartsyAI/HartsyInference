using System.IO;
using System.Linq;
using System.Text;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.ModelAssets.Tokenizers.Tests;

/// <summary>Tests the BERT WordPiece tokenizer (Bark's text front-end) with a tiny in-memory vocab — raw ids,
/// no special tokens. The real bert-base-multilingual-cased vocab.txt is runtime data the caller supplies.</summary>
public sealed class BertWordPieceTokenizerTests
{
    // vocab.txt: one token per line; id == line index (0-based).
    private const string Vocab = "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\nthe\ncat\n";

    private static BertWordPieceTokenizer Make()
        => new(new MemoryStream(Encoding.UTF8.GetBytes(Vocab)), lowerCase: false);

    [Fact]
    public void EncodeRaw_KnownWords_NoSpecialTokens()
    {
        using BertWordPieceTokenizer tok = Make();
        int[] ids = [.. tok.EncodeRaw("hello world")];
        Assert.Equal([5, 6], ids); // hello=5, world=6; no [CLS]/[SEP] wrapping
    }

    [Fact]
    public void EncodeRaw_Unknown_FallsBackToUnk()
    {
        using BertWordPieceTokenizer tok = Make();
        int[] ids = [.. tok.EncodeRaw("zzzz")];
        Assert.Equal([1], ids); // [UNK] = 1
    }
}
