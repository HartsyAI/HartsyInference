using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>RecursiveSI/flint through the official GGUF header only: the metadata against the checkpoint's <c>config.json</c>, and the GGUF tokenizer and chat template against ids from the checkpoint's own <c>tokenizer.json</c> and chat template through Hugging Face transformers 5.22.1. The tensors are never read, so the first 150 MB of RecursiveSI/flint-int4-gguf is enough. Skips when absent (set FLINT_GGUF_PATH).</summary>
public sealed class FlintGgufHeaderTests(ITestOutputHelper output)
{
    public static TheoryData<string, int[]> Samples() => new()
    {
        { "The capital of France is Paris. The capital of Germany is",
            [630, 6963, 297, 9865, 332, 12834, 13, 450, 6963, 297, 10444, 332] },
        { "Hello, world! 12345 + 678 = 813.",
            [9479, 11, 1571, 0, 220, 4051, 1362, 541, 220, 14035, 311, 220, 16918, 13] },
        { "def f(x):\n    return x * 2\n",
            [1182, 280, 1940, 1080, 279, 597, 782, 461, 220, 17, 198] },
        { "\u4e2d\u6587\u6d4b\u8bd5\uff1a\u4f60\u597d\uff0c\u4e16\u754c\u3002  Multiple   spaces\tand tabs.",
            [50749, 22139, 2925, 131780, 852, 21367, 1050, 220, 20190, 256, 8802, 38957, 27195, 13] },
        { "na\u00efve caf\u00e9 \u2014 \u201cquotes\u201d \ud83d\ude00",
            [2876, 69556, 59029, 2911, 1244, 66863, 975, 145593] },
        { "import numpy as np\n\nclass A:\n\tdef __init__(self):\n\t\tpass\n",
            [946, 8768, 389, 2775, 284, 1373, 344, 537, 10798, 2409, 2180, 5082, 1473, 1080, 197, 61300, 198] },
    };

    private static string? GgufPath()
    {
        string? path = Environment.GetEnvironmentVariable("FLINT_GGUF_PATH");
        return !string.IsNullOrEmpty(path) && File.Exists(path) ? path : null;
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void MetadataMatchesTheCheckpointConfig()
    {
        if (GgufPath() is not string path) { output.WriteLine("SKIPPED: set FLINT_GGUF_PATH to the Flint GGUF (the header is enough)."); return; }
        using GgufLoader loader = new();
        loader.Load(path);
        GgufMetadata m = loader.Metadata;
        Assert.Equal("flint", m.GetString("general.architecture"));
        Assert.Equal(49u, m.GetUInt32("flint.block_count"));
        Assert.Equal(2048u, m.GetUInt32("flint.embedding_length"));
        Assert.Equal(6144u, m.GetUInt32("flint.feed_forward_length"));
        Assert.Equal(32u, m.GetUInt32("flint.attention.head_count"));
        Assert.Equal(4u, m.GetUInt32("flint.attention.head_count_kv"));
        Assert.Equal(128u, m.GetUInt32("flint.attention.key_length"));
        Assert.Equal(10_000f, m.GetFloat32("flint.rope.freq_base"));
        Assert.Equal(256u, m.GetUInt32("flint.expert_count"));
        Assert.Equal(6u, m.GetUInt32("flint.expert_used_count"));
        Assert.Equal(512u, m.GetUInt32("flint.expert_feed_forward_length"));
        Assert.Equal(1024u, m.GetUInt32("flint.expert_shared_feed_forward_length"));
        Assert.Equal(2u, m.GetUInt32("flint.leading_dense_block_count"));
        Assert.Equal(2u, m.GetUInt32("flint.expert_gating_func"));
        Assert.True(m.GetBool("flint.expert_weights_norm"));
        Assert.Equal(513u, m.GetUInt32("flint.attention.sliding_window"));
        bool[] sliding = m.GetBoolArray("flint.attention.sliding_window_pattern")!;
        Assert.Equal(49, sliding.Length);
        // config.json layer_types: the two leading dense layers, layer 48 and every fourth layer from the sixth are full attention.
        for (int i = 0; i < 49; i++)
            Assert.Equal(!(i is 0 or 1 or 48 || (i >= 5 && (i - 5) % 4 == 0)), sliding[i]);
        Assert.Equal(150144, m.GetStringArray("tokenizer.ggml.tokens")!.Length);
    }

    [Theory]
    [MemberData(nameof(Samples))]
    [Trait("Category", "Integration")]
    public void TokenizerEncodesLikeTheCheckpointsOwnTokenizer(string text, int[] expected)
    {
        if (GgufPath() is not string path) { output.WriteLine("SKIPPED: set FLINT_GGUF_PATH to the Flint GGUF (the header is enough)."); return; }
        using GgufLoader loader = new();
        loader.Load(path);
        ILlmTokenizer tokenizer = GgufLanguageModel.BuildTokenizer(loader.Metadata);
        int[] ids = tokenizer.EncodeOrdinary(text);
        output.WriteLine($"{ids.Length} ids (reference {expected.Length})");
        Assert.Equal(expected, ids);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void StopIdsIncludeEveryEndOfGenerationToken()
    {
        if (GgufPath() is not string path) { output.WriteLine("SKIPPED: set FLINT_GGUF_PATH to the Flint GGUF (the header is enough)."); return; }
        using GgufLoader loader = new();
        loader.Load(path);
        ILlmTokenizer tokenizer = GgufLanguageModel.BuildTokenizer(loader.Metadata);
        foreach (int id in loader.Metadata.GetIntArray("tokenizer.ggml.eog_token_ids")!)
            Assert.Contains(id, tokenizer.StopIds);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ChatTemplateRendersLikeTransformers()
    {
        if (GgufPath() is not string path) { output.WriteLine("SKIPPED: set FLINT_GGUF_PATH to the Flint GGUF (the header is enough)."); return; }
        using GgufLoader loader = new();
        loader.Load(path);
        ILlmTokenizer tokenizer = GgufLanguageModel.BuildTokenizer(loader.Metadata);
        IChatTemplate template = GgufLanguageModel.BuildTemplate(loader.Metadata, tokenizer);
        int[] single = template.Encode(tokenizer, [new ChatMessage("user", "What is the capital of France? Answer with one word.")], true);
        output.WriteLine(tokenizer.Decode(single));
        Assert.Equal(ChatIds, single);
        int[] multi = template.Encode(tokenizer,
            [new ChatMessage("system", "You are terse."), new ChatMessage("user", "Hi"), new ChatMessage("assistant", "Hello."), new ChatMessage("user", "Bye")], true);
        Assert.Equal(MultiTurnIds, multi);
    }

    // transformers apply_chat_template(add_generation_prompt=True) then encode.
    private static readonly int[] ChatIds = [150003, 2387, 332, 269, 6963, 297, 9865, 30, 13755, 386, 703, 1867, 13, 150004];
    private static readonly int[] MultiTurnIds = [150002, 2356, 422, 51598, 13, 150003, 10340, 150004, 9479, 13, 150003, 85711, 150004];
}
