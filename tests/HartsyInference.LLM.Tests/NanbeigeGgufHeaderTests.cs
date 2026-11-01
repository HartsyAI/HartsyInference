using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>Nanbeige4.3-33B (a plain Llama architecture) through its community GGUF header only (reference ids: <c>tests/python-reference/dump_nanbeige_tokenizer_reference.py</c>): the metadata against the checkpoint's <c>config.json</c>, and the GGUF tokenizer and chat template against ids from the checkpoint's own <c>tokenizer.json</c> and chat template through Hugging Face transformers. The tensors are never read, so the first 100 MB of any quant of bartowski/Nanbeige_Nanbeige4.3-33B-GGUF is enough. Skips when absent (set NANBEIGE_GGUF_PATH).</summary>
public sealed class NanbeigeGgufHeaderTests(ITestOutputHelper output)
{
    public static TheoryData<string, int[]> Samples() => new()
    {
        { "The capital of France is Paris. The capital of Germany is",
            [363, 5463, 290, 7914, 322, 9965, 152361, 363, 5463, 290, 7582, 322] },
        { "Hello, world! 12345 + 678 = 813.",
            [23877, 152363, 1427, 152518, 152343, 152367, 152375, 152385, 152395, 152394, 670, 152343, 152408, 152413, 152411, 374, 152343, 152411, 152367, 152385, 152361] },
        { "def f(x):\n    return x * 2\n",
            [862, 283, 152374, 152381, 1123, 13, 264, 1590, 1056, 753, 152343, 152375, 13] },
        { "\u4e2d\u6587\u6d4b\u8bd5\uff1a\u4f60\u597d\uff0c\u4e16\u754c\u3002  Multiple   spaces\tand tabs.",
            [66084, 3749, 152473, 22779, 152372, 1716, 152389, 259, 43718, 259, 8347, 12, 407, 39954, 152361] },
        { "na\u00efve caf\u00e9 \u2014 \u201cquotes\u201d \ud83d\ude00",
            [77485, 84959, 2462, 621, 98704, 152442, 152343, 158621] },
    };

    private static string? GgufPath()
    {
        string? path = Environment.GetEnvironmentVariable("NANBEIGE_GGUF_PATH");
        return !string.IsNullOrEmpty(path) && File.Exists(path) ? path : null;
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void MetadataMatchesTheCheckpointConfig()
    {
        if (GgufPath() is not string path) { output.WriteLine("SKIPPED: set NANBEIGE_GGUF_PATH to a Nanbeige4.3-33B GGUF (the header is enough)."); return; }
        using GgufLoader loader = new();
        loader.Load(path);
        GgufMetadata m = loader.Metadata;
        Assert.Equal("llama", m.GetString("general.architecture"));
        Assert.Equal(52u, m.GetUInt32("llama.block_count"));
        Assert.Equal(7168u, m.GetUInt32("llama.embedding_length"));
        Assert.Equal(22016u, m.GetUInt32("llama.feed_forward_length"));
        Assert.Equal(56u, m.GetUInt32("llama.attention.head_count"));
        Assert.Equal(8u, m.GetUInt32("llama.attention.head_count_kv"));
        Assert.Equal(120_000_000f, m.GetFloat32("llama.rope.freq_base"));
        Assert.Equal(1e-5f, m.GetFloat32("llama.attention.layer_norm_rms_epsilon"));
        Assert.Equal(166912, m.GetStringArray("tokenizer.ggml.tokens")!.Length);
    }

    [Theory]
    [MemberData(nameof(Samples))]
    [Trait("Category", "Integration")]
    public void TokenizerEncodesLikeTheCheckpointsOwnTokenizer(string text, int[] expected)
    {
        if (GgufPath() is not string path) { output.WriteLine("SKIPPED: set NANBEIGE_GGUF_PATH to a Nanbeige4.3-33B GGUF (the header is enough)."); return; }
        using GgufLoader loader = new();
        loader.Load(path);
        ILlmTokenizer tokenizer = GgufLanguageModel.BuildTokenizer(loader.Metadata);
        int[] ids = tokenizer.EncodeOrdinary(text);
        output.WriteLine($"{ids.Length} ids (reference {expected.Length})");
        Assert.Equal(expected, ids);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ChatTemplateRendersLikeTransformers()
    {
        if (GgufPath() is not string path) { output.WriteLine("SKIPPED: set NANBEIGE_GGUF_PATH to a Nanbeige4.3-33B GGUF (the header is enough)."); return; }
        using GgufLoader loader = new();
        loader.Load(path);
        ILlmTokenizer tokenizer = GgufLanguageModel.BuildTokenizer(loader.Metadata);
        IChatTemplate template = GgufLanguageModel.BuildTemplate(loader.Metadata, tokenizer);
        int[] ids = template.Encode(tokenizer, [new ChatMessage("user", "What is the capital of France? Answer with one word.")], true);
        output.WriteLine(tokenizer.Decode(ids));
        Assert.Equal(ChatIds, ids);
    }

    // transformers apply_chat_template(add_generation_prompt=True) then encode: the default system turn, the user turn, and an open <think>.
    private static readonly int[] ChatIds = [166100, 8481, 13, 18215, 19433, 154675, 152372, 21863, 152672, 57437, 162001, 46372, 152590, 5004, 15825, 152435, 3237, 3942, 152389, 166101, 13, 166100, 2714, 13, 3512, 322, 269, 5463, 290, 7914, 152470, 14770, 358, 609, 3670, 152361, 166101, 13, 166100, 66354, 13, 166103, 13];
}
