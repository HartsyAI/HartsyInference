using HartsyInference.Audio.Models.Orpheus;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Checkpoint-free tests for Orpheus: the Llama-3.2-3B preset shape and the flat-stream → SNAC
/// 3-layer redistribution (the one piece of pure index bookkeeping that must be exactly right).</summary>
public sealed class OrpheusTests
{
    [Fact]
    public void Orpheus3B_PresetMatchesLlama32_3B()
    {
        OrpheusConfig c = OrpheusConfig.Orpheus3B;
        Assert.Equal(3_072, c.Llm.HiddenSize);
        Assert.Equal(28, c.Llm.NumHiddenLayers);
        Assert.Equal(24, c.Llm.NumAttentionHeads);
        Assert.Equal(8, c.Llm.NumKeyValueHeads);
        Assert.Equal(128, c.Llm.HeadDim);               // 3072 / 24
        Assert.False(c.Llm.AttentionBias);              // Llama has no attn bias
        Assert.True(c.Llm.TieWordEmbeddings);
        Assert.Equal(500_000f, c.Llm.RopeTheta);
        Assert.Equal(7, c.TokensPerFrame);
        Assert.Equal(3, c.Codec.NCodebooks);            // SNAC 24 kHz speech variant
        Assert.Equal(4_096, c.CodebookSize);
        Assert.Equal(128_258, c.EndOfSpeech);
    }

    [Fact]
    public void Extract_CropsBeforeLastCodeStart_AndTrimsToFrameMultiple()
    {
        // junk, CODE_START, then 9 codes (one full 7-group + 2 leftovers) with an EOS in the middle.
        int codeStart = 128_257, eos = 128_258;
        List<int> gen = [42, 7, codeStart, 200, 201, 202, eos, 203, 204, 205, 206, 207, 208];
        int[] codes = OrpheusCodeFrames.ExtractAudioCodes(gen, codeStart, eos, 7);
        // After last CODE_START: [200,201,202,(eos dropped),203,204,205,206,207,208] = 9 → trim to 7.
        Assert.Equal(7, codes.Length);
        Assert.Equal(new[] { 200, 201, 202, 203, 204, 205, 206 }, codes);
    }

    [Fact]
    public void Redistribute_SplitsSevenIntoHierarchy_WithPerPositionOffsets()
    {
        int baseId = 128_266, cb = 4_096;
        // Build one super-frame whose dequantized codes are a known [10,11,12,13,14,15,16].
        int[] codes = new int[7];
        int[] want = [10, 11, 12, 13, 14, 15, 16];
        for (int p = 0; p < 7; p++) codes[p] = want[p] + baseId + p * cb;

        (int[] l1, int[] l2, int[] l3) = OrpheusCodeFrames.Redistribute(codes, baseId, cb);
        Assert.Equal(new[] { 10 }, l1);                 // pos 0
        Assert.Equal(new[] { 11, 14 }, l2);             // pos 1, 4
        Assert.Equal(new[] { 12, 13, 15, 16 }, l3);     // pos 2, 3, 5, 6
    }

    [Fact]
    public void Redistribute_LayerLengthsAreOneTwoFourPerGroup()
    {
        int[] codes = new int[7 * 3];                   // 3 super-frames
        (int[] l1, int[] l2, int[] l3) = OrpheusCodeFrames.Redistribute(codes, 128_266, 4_096);
        Assert.Equal(3, l1.Length);
        Assert.Equal(6, l2.Length);
        Assert.Equal(12, l3.Length);
    }

    [Fact]
    public void Dequant_ClampsOutOfRangeCodesIntoCodebook()
    {
        // A position-0 token far below the base → negative code → clamped to 0.
        int[] codes = new int[7];                       // all zero ids → very negative codes
        (int[] l1, _, _) = OrpheusCodeFrames.Redistribute(codes, 128_266, 4_096);
        Assert.Equal(0, l1[0]);
    }
}
