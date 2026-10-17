using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Looped;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.LLM.Tests.Looped;

/// <summary>Reading llama.cpp's <c>dory.*</c> metadata into a <see cref="LoopedTransformerConfig"/>: the layer pattern, the two-base RoPE profile, the
/// block sizes and the cache mode, and the refusals for a file whose metadata disagrees with itself.</summary>
public sealed class LoopedConfigFactoryTests
{
    private static GgufMetadata Metadata(string pattern = "S-*-S-", int[]? profile = null, string? mode = "per_loop",
        uint input = 2, uint recurrent = 2, uint output = 2)
    {
        GgufMetadata m = new();
        m.Add("general.architecture", "dory");
        m.Add("dory.block_count", (uint)pattern.Length);
        m.Add("dory.embedding_length", 16u);
        m.Add("dory.attention.head_count", 2u);
        m.Add("dory.attention.head_count_kv", 1u);
        m.Add("dory.feed_forward_length", 24u);
        m.Add("dory.attention.key_length", 8u);
        m.Add("dory.attention.value_length", 8u);
        m.Add("dory.rope.dimension_count", 8u);
        m.Add("dory.context_length", 4096u);
        m.Add("dory.rope.freq_base", 10_000_000f);
        m.Add("dory.rope.freq_base_2", 10_000f);
        m.Add("dory.attention.sliding_window", 1024u);
        m.Add("dory.input_layer_count", input);
        m.Add("dory.recurrent_layer_count", recurrent);
        m.Add("dory.output_layer_count", output);
        m.Add("dory.num_loops", 2u);
        m.Add("dory.layer_pattern", pattern);
        m.Add("dory.rope.profile", profile ?? [1, 1, 1, 1, 2, 1]);
        if (mode is not null) m.Add("dory.recurrent_kv_cache_mode", mode);
        m.Add("dory.init_std", 0.02f);
        m.Add("dory.alpha_init", 0.05f);
        m.Add("dory.gate_init", 4f);
        return m;
    }

    private static Dictionary<string, Tensor> Weights() =>
        new() { ["model.embed_tokens.weight"] = new Tensor(new TensorShape(20, 16), DType.F32) };

    [Fact]
    public void ReadsBlocksPatternProfileAndMode()
    {
        LoopedTransformerConfig cfg = LoopedConfigFactory.FromGguf(Metadata(), Weights());

        Assert.Equal("dory", cfg.Architecture);
        Assert.Equal((2, 2, 2, 2), (cfg.InputLayers, cfg.RecurrentLayers, cfg.OutputLayers, cfg.Loops));
        Assert.Equal(4, cfg.EffectiveMaxLoops);
        Assert.Equal(20, cfg.VocabSize);
        Assert.Equal(4096, cfg.MaxPositionEmbeddings);
        Assert.Equal([LoopedLayerKind.SlidingAttention, LoopedLayerKind.Mlp, LoopedLayerKind.FullAttention, LoopedLayerKind.Mlp,
            LoopedLayerKind.SlidingAttention, LoopedLayerKind.Mlp], cfg.LayerKinds);
        // profile 1 selects rope.freq_base, profile 2 rope.freq_base_2
        Assert.Equal(10_000_000f, cfg.RopeThetas[0]);
        Assert.Equal(10_000f, cfg.RopeThetas[4]);
        Assert.Equal(1024, cfg.SlidingWindow);
        Assert.Equal(LoopedKvCacheMode.PerLoop, cfg.KvCacheMode);
        Assert.Equal((0.02f, 0.05f, 4f), (cfg.InitStd, cfg.AlphaInit, cfg.GateInit));
    }

    [Fact]
    public void LastLoopMode_AndAbsentMode_AreRead()
    {
        Assert.Equal(LoopedKvCacheMode.LastLoop, LoopedConfigFactory.FromGguf(Metadata(mode: "last_loop"), Weights()).KvCacheMode);
        Assert.Equal(LoopedKvCacheMode.PerLoop, LoopedConfigFactory.FromGguf(Metadata(mode: null), Weights()).KvCacheMode);
        Assert.Throws<NotSupportedException>(() => LoopedConfigFactory.FromGguf(Metadata(mode: "every_other"), Weights()));
    }

    [Fact]
    public void RefusesDisagreeingMetadata()
    {
        Assert.Throws<ArgumentException>(() => LoopedConfigFactory.FromGguf(Metadata(pattern: "S-X-S-"), Weights()));
        Assert.Throws<ArgumentException>(() => LoopedConfigFactory.FromGguf(Metadata(profile: [1, 1, 3, 1, 2, 1]), Weights()));
        Assert.Throws<ArgumentException>(() => LoopedConfigFactory.FromGguf(Metadata(input: 3), Weights()));
        Assert.Throws<ArgumentException>(() => LoopedConfigFactory.FromGguf(Metadata(profile: [1, 1]), Weights()));
    }

    [Fact]
    public void OtherArchitectures_AreNotLooped()
    {
        GgufMetadata m = new();
        m.Add("general.architecture", "llama");
        Assert.Throws<ArgumentException>(() => LoopedConfigFactory.FromGguf(m, Weights()));
        Assert.False(LoopedLanguageModel.IsLoopedArchitecture("llama"));
        Assert.True(LoopedLanguageModel.IsLoopedArchitecture("dory"));
    }
}
