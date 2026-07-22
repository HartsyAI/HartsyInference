using System.Text;
using Xunit;

namespace HartsyInference.ModelAssets.Tokenizers.Tests;

/// <summary>Structural tests for the ERNIE-Image tokenizer.json loader. A synthetic byte-level BPE tokenizer.json (256 byte tokens + a few merges, in the exact HF-tokenizers serialization) is written to a temp file so these always run; numeric parity with Baidu's real tokenizer.json is validation-gated on a checkpoint download (set <c>ERNIE_TOKENIZER_JSON</c> for the optional real-file smoke test).</summary>
public sealed class ErnieTokenizerTests : IDisposable
{
    private readonly string _tokenizerJsonPath;

    public ErnieTokenizerTests()
    {
        _tokenizerJsonPath = Path.Combine(Path.GetTempPath(), $"ernie_tok_{Guid.NewGuid():N}.json");
        File.WriteAllText(_tokenizerJsonPath, BuildSyntheticTokenizerJson(), new UTF8Encoding(false));
    }

    public void Dispose()
    {
        try { File.Delete(_tokenizerJsonPath); } catch { /* temp-file cleanup is best-effort */ }
    }

    [Fact]
    public void Encode_PlacesBosFirstAndEosLast()
    {
        using ErnieTokenizer tok = new(_tokenizerJsonPath);
        int[] ids = tok.Encode("hello world");

        Assert.Equal(ErnieTokenizer.BosId, ids[0]);
        Assert.Equal(ErnieTokenizer.EosId, ids[^1]);
        Assert.True(ids.Length > 2, "Expected real text tokens between BOS and EOS");
    }

    [Fact]
    public void Encode_NoBosNoEos_ReturnsOnlyTextTokens()
    {
        using ErnieTokenizer tok = new(_tokenizerJsonPath);
        int[] full = tok.Encode("hello world");
        int[] bare = tok.Encode("hello world", addBos: false, appendEos: false);

        Assert.Equal(full.Length - 2, bare.Length);
        Assert.DoesNotContain(ErnieTokenizer.BosId, bare);
        Assert.DoesNotContain(ErnieTokenizer.EosId, bare);
        Assert.Equal(full[1..^1], bare);
    }

    [Fact]
    public void Encode_TokenIdsWithinVocabRange()
    {
        using ErnieTokenizer tok = new(_tokenizerJsonPath);
        int[] bare = tok.Encode("the cat sat on the mat", addBos: false, appendEos: false);

        // Synthetic vocab ids occupy [TokenIdBase, TokenIdBase + VocabSize); specials sit below.
        Assert.All(bare, id => Assert.InRange(id, TokenIdBase, TokenIdBase + tok.VocabSize - 1));
    }

    [Fact]
    public void Encode_LongerPromptYieldsMoreTokens()
    {
        using ErnieTokenizer tok = new(_tokenizerJsonPath);
        int[] shortIds = tok.Encode("cat");
        int[] longIds = tok.Encode("a magnificent dragon flying over a castle at sunset");
        Assert.True(longIds.Length > shortIds.Length);
    }

    [Fact]
    public void EncodeRaw_AppliesMergesFromBothSerializationForms()
    {
        using ErnieTokenizer tok = new(_tokenizerJsonPath);
        // "he" merge is serialized as a legacy string, "ll" as a new-style array — both must load.
        IReadOnlyList<int> ids = tok.EncodeRaw("hell");
        Assert.True(ids.Count < 4, $"Merges were not applied: 'hell' produced {ids.Count} tokens");
    }

    [Fact]
    public void Constructor_MissingFile_Throws()
    {
        Assert.Throws<FileNotFoundException>(() => new ErnieTokenizer("/nonexistent/tokenizer.json"));
    }

    [Fact]
    public void Encode_RealTokenizerJson_WhenEnvVarSet()
    {
        string? path = Environment.GetEnvironmentVariable("ERNIE_TOKENIZER_JSON");
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
            return; // validation-gated: needs the real baidu/ERNIE-Image tokenizer.json

        using ErnieTokenizer tok = new(path);
        int[] ids = tok.Encode("A photograph of an astronaut riding a horse");
        Assert.Equal(ErnieTokenizer.BosId, ids[0]);
        Assert.Equal(ErnieTokenizer.EosId, ids[^1]);
        Assert.True(ids.Length > 2);
        Assert.All(ids, id => Assert.InRange(id, 0, 131071));
    }

    // ── Synthetic tokenizer.json ───────────────────────────────────────────

    /// <summary>First id assigned to byte-level tokens — leaves room for the Mistral special ids (0/1/2/11).</summary>
    private const int TokenIdBase = 12;

    /// <summary>Builds a minimal HF-tokenizers BPE tokenizer.json: 256 byte-level tokens (GPT-2 byte→unicode mapping), two merged tokens, and merges in both the legacy string and new array forms.</summary>
    private static string BuildSyntheticTokenizerJson()
    {
        StringBuilder vocab = new();
        int id = TokenIdBase;
        for (int b = 0; b < 256; b++)
        {
            if (vocab.Length > 0) vocab.Append(',');
            vocab.Append('"').Append(JsonEscape(ByteToUnicode(b))).Append("\":").Append(id++);
        }
        vocab.Append(",\"he\":").Append(id++);
        vocab.Append(",\"ll\":").Append(id);

        return
            "{\"version\":\"1.0\"," +
            "\"added_tokens\":[" +
            "{\"id\":0,\"content\":\"<unk>\",\"special\":true}," +
            "{\"id\":1,\"content\":\"<s>\",\"special\":true}," +
            "{\"id\":2,\"content\":\"</s>\",\"special\":true}," +
            "{\"id\":11,\"content\":\"<pad>\",\"special\":true}]," +
            "\"model\":{\"type\":\"BPE\",\"vocab\":{" + vocab + "}," +
            "\"merges\":[\"h e\",[\"l\",\"l\"]]}}";
    }

    /// <summary>GPT-2 byte→unicode mapping: printable Latin-1 ranges map to themselves, everything else to U+0100+n.</summary>
    private static string ByteToUnicode(int b)
    {
        bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        if (printable)
            return ((char)b).ToString();

        int shifted = 0x100;
        for (int i = 0; i < b; i++)
        {
            bool p = (i >= '!' && i <= '~') || (i >= 0xA1 && i <= 0xAC) || (i >= 0xAE && i <= 0xFF);
            if (!p) shifted++;
        }
        return ((char)shifted).ToString();
    }

    private static string JsonEscape(string s)
    {
        StringBuilder sb = new(s.Length);
        foreach (char c in s)
        {
            if (c == '"' || c == '\\') sb.Append('\\').Append(c);
            else if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4"));
            else sb.Append(c);
        }
        return sb.ToString();
    }
}
