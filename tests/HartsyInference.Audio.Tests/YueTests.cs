using HartsyInference.Audio.Models.Music;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Checkpoint-free config test for YuE — confirms the Stage-1 LM is set up to match the real
/// <c>m-a-p/YuE-s1-7B-anneal-en-cot</c> config.json: a LLaMA-2-7B body reusing the (bias-off) Qwen2
/// transformer, but with GQA (4 KV heads) and the YuE-extended 83968 vocab. These two dims were verified
/// against the real checkpoint header (k/v_proj = [512, 4096]; embed = [83968, 4096]) and a teacher-forced
/// logits parity run (corr 1.0, argmax 8/8 match) — see PARITY_VERIFICATION.md.</summary>
public sealed class YueTests
{
    [Fact]
    public void Config_V1_Stage1_IsLlama2_7B()
    {
        YueConfig c = YueConfig.V1;
        Assert.Equal(4_096, c.Stage1.HiddenSize);
        Assert.Equal(32, c.Stage1.NumHiddenLayers);
        Assert.Equal(32, c.Stage1.NumAttentionHeads);
        Assert.Equal(4, c.Stage1.NumKeyValueHeads);      // GQA: real config.json num_key_value_heads = 4
        Assert.Equal(83_968, c.Stage1.VocabSize);        // YuE-extended vocab (real embed = [83968, 4096])
        Assert.False(c.Stage1.AttentionBias);            // Llama = no bias
        Assert.Equal(11_008, c.Stage1.IntermediateSize);
        Assert.Equal(8, c.NumCodebooks);
        Assert.Equal(16_000, c.SampleRate);
        Assert.Equal(50, c.FrameRateHz);
        Assert.Equal(1.1f, c.RepetitionPenalty);         // mandatory per YuE
        // Vocal track cb0 starts at <xcodec/0/0> = 45334 (verified against the real mm tokenizer.model).
        Assert.Equal(45_334, c.VocalTokenBase);
    }
}
