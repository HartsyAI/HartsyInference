using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Kimi;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>The Kimi K3.1 tokenizer on the published 163,584-rank <c>tiktoken.model</c>. Skips unless <c>HARTSY_KIMI_TOKENIZER_DIR</c> names the directory
/// that holds it. The expected ids were computed by a reference re-implementation of tiktoken's BPE with the published pattern (the tiktoken package
/// itself was not installed), so this checks the engine against that reference, not against tiktoken.</summary>
[Trait("Category", "Integration")]
public sealed class KimiK31RealVocabularyTests
{
    private const int BosId = 163584, EosId = 163586;

    // Plain text, ordinary encode (no special handling), from the reference.
    private static readonly (string Text, int[] Ids)[] TextIds =
    [
        ("Hello world", [19180, 2695]),
        ("Don't stop", [34838, 6156]),
        ("x = 12345678", [87, 327, 220, 6694, 12972, 4447]),
        ("中文和English混合", [16717, 488, 44372, 13935]),
        ("emoji 😀 done!!", [33120, 70005, 222, 4179, 4605]),
        ("price: $3.50 (approx.)", [18493, 25, 701, 18, 13, 1643, 347, 65158, 10645]),
    ];

    // Token ids of the default single-user prompt: the effort turn, the user turn and the thinking generation prompt.
    private static readonly int[] DefaultPromptIds =
    [
        163587, 2778, 6244, 878, 14062, 1, 1798, 878, 130400, 59991, 470, 1, 163589, 63, 130400, 123074, 470, 63, 28560, 418, 1632, 2455, 308, 2704,
        306, 651, 9545, 8491, 347, 2976, 3411, 276, 4503, 8491, 904, 9141, 4661, 3867, 1268, 1332, 5713, 1268, 50004, 5713, 1268, 19421, 5713, 316,
        1268, 5354, 16518, 10048, 276, 2403, 387, 42070, 472, 1268, 130400, 123074, 470, 105281, 11369, 163588, 2778, 163589, 163586, 163587, 2778,
        6244, 878, 2482, 1, 163589, 19180, 11, 24980, 72, 0, 163588, 2778, 163589, 163586, 163587, 2778, 6244, 878, 69702, 1, 163589, 163587, 39964,
        163589,
    ];

    [KimiTokenizerDirFact]
    public void Real_ranks_encode_the_reference_ids_and_decode_back()
    {
        ILlmTokenizer tokenizer = Load();

        Assert.Equal(BosId, tokenizer.BosId);
        Assert.Equal(EosId, tokenizer.EosId);
        foreach ((string text, int[] ids) in TextIds)
        {
            Assert.Equal(ids, tokenizer.EncodeOrdinary(text));
            Assert.Equal(text, tokenizer.Decode(ids));
        }
    }

    [KimiTokenizerDirFact]
    public void Real_chat_prompt_matches_the_reference_ids()
    {
        ILlmTokenizer tokenizer = Load();

        int[] ids = new KimiK31ChatTemplate().Encode(tokenizer, [ChatMessage.User("Hello, Kimi!")], addGenerationPrompt: true, enableThinking: true);

        Assert.Equal(DefaultPromptIds, ids);
    }

    private static ILlmTokenizer Load()
    {
        string dir = Environment.GetEnvironmentVariable(KimiTokenizerDirFactAttribute.EnvironmentVariable)!;
        return KimiK31Tokenizer.Load(dir, [EosId], BosId, EosId);
    }
}
