using System.Globalization;

namespace HartsyInference.ModelAssets.Checkpoints;

/// <summary>DwarfStar's <c>deepseek41</c> GGUF names (<c>blk.N.*</c>, <c>token_embd</c>, <c>output</c>) to the canonical names.</summary>
/// <remarks>
/// <para>Names come from the real Q2 header (antirez/deepseek-v4.1-flash-gguf @ dd8a266f, 1046 tensors). Routed experts are stored fused, one 3-D tensor per projection per layer, so they map to <c>layers.N.ffn.experts.{w1,w2,w3}.weight</c> with no expert index; a consumer slices the expert axis itself.</para>
/// <para>The Engram table <c>blk.N.engram_embd.weight</c> is <c>I8</c> with GGUF dims <c>[264, rows]</c> (row-major <c>[rows, 264]</c>): each row is the 256 e4m3 payload bytes then the 8 e8m0 scale bytes, where the official checkpoint keeps them as separate <c>embed.weight</c> and <c>embed.scale</c> tensors. It maps to <c>embed.weight</c> and there is no <c>embed.scale</c> counterpart.</para>
/// <para>The draft heads are left out of this producer's files, so any <c>blk.N</c> at or past the backbone depth and any <c>mtp.*</c> key maps to null.</para></remarks>
public sealed class DwarfStarV41KeyMapper : IHfKeyMapper
{
    /// <summary>Bytes per Engram row: <see cref="EngramPayloadBytes"/> e4m3 values then <see cref="EngramScaleBytes"/> e8m0 scales.</summary>
    public const int EngramRowBytes = EngramPayloadBytes + EngramScaleBytes;

    /// <summary>Leading e4m3 payload bytes of one Engram row.</summary>
    public const int EngramPayloadBytes = 256;

    /// <summary>Trailing e8m0 scale bytes of one Engram row.</summary>
    public const int EngramScaleBytes = 8;

    private const string BlockPrefix = "blk.";

    private static readonly Dictionary<string, string> Globals = new(StringComparer.Ordinal)
    {
        ["token_embd.weight"] = "embed.weight",
        ["output_norm.weight"] = "norm.weight",
        ["output.weight"] = "head.weight",
    };

    private static readonly Dictionary<string, string> BlockSuffixes = new(StringComparer.Ordinal)
    {
        ["hc_attn_fn.weight"] = "hc_attn_fn",
        ["hc_attn_base.weight"] = "hc_attn_base",
        ["hc_attn_scale.weight"] = "hc_attn_scale",
        ["hc_ffn_fn.weight"] = "hc_ffn_fn",
        ["hc_ffn_base.weight"] = "hc_ffn_base",
        ["hc_ffn_scale.weight"] = "hc_ffn_scale",
        ["attn_norm.weight"] = "attn_norm.weight",
        ["ffn_norm.weight"] = "ffn_norm.weight",
        ["attn_sinks.weight"] = "attn.attn_sink",
        ["attn_q_a.weight"] = "attn.wq_a.weight",
        ["attn_q_b.weight"] = "attn.wq_b.weight",
        ["attn_q_a_norm.weight"] = "attn.q_norm.weight",
        ["attn_kv.weight"] = "attn.wkv.weight",
        ["attn_kv_a_norm.weight"] = "attn.kv_norm.weight",
        ["attn_output_a.weight"] = "attn.wo_a.weight",
        ["attn_output_b.weight"] = "attn.wo_b.weight",
        ["attn_compressor_kv.weight"] = "attn.compressor.wkv.weight",
        ["attn_compressor_gate.weight"] = "attn.compressor.wgate.weight",
        ["attn_compressor_norm.weight"] = "attn.compressor.norm.weight",
        ["indexer.attn_k.weight"] = "attn.indexer.wk.weight",
        ["indexer.attn_q_b.weight"] = "attn.indexer.wq_b.weight",
        ["indexer.k_norm.weight"] = "attn.indexer.k_norm.weight",
        ["indexer.proj.weight"] = "attn.indexer.weights_proj.weight",
        ["ffn_gate_inp.weight"] = "ffn.gate.weight",
        ["exp_probs_b.bias"] = "ffn.gate.bias",
        ["exp_probs_b_vl.bias"] = "ffn.gate.bias_vl",
        ["ffn_gate_shexp.weight"] = "ffn.shared_experts.w1.weight",
        ["ffn_down_shexp.weight"] = "ffn.shared_experts.w2.weight",
        ["ffn_up_shexp.weight"] = "ffn.shared_experts.w3.weight",
        ["ffn_gate_exps.weight"] = "ffn.experts.w1.weight",
        ["ffn_down_exps.weight"] = "ffn.experts.w2.weight",
        ["ffn_up_exps.weight"] = "ffn.experts.w3.weight",
        ["engram_q_norm.weight"] = "engram.q_weight",
        ["engram_k_norm.weight"] = "engram.k_weight",
        ["engram_kv.weight"] = "engram.wkv.weight",
        ["engram_embd.weight"] = "engram.embed.weight",
    };

    private readonly int _backboneLayers;

    /// <summary>Creates a mapper for a checkpoint with <paramref name="backboneLayers"/> backbone layers (the model's <c>num_hidden_layers</c>).</summary>
    public DwarfStarV41KeyMapper(int backboneLayers = 40)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(backboneLayers, 1);
        _backboneLayers = backboneLayers;
    }

    /// <inheritdoc/>
    public IReadOnlySet<string> StrippedComponents { get; } = new HashSet<string>(StringComparer.Ordinal) { "mtp" };

    /// <inheritdoc/>
    public string? MapToCanonical(string sourceKey)
    {
        ArgumentNullException.ThrowIfNull(sourceKey);
        if (Globals.TryGetValue(sourceKey, out string? global))
            return global;
        if (!sourceKey.StartsWith(BlockPrefix, StringComparison.Ordinal))
            return null;

        int dot = sourceKey.IndexOf('.', BlockPrefix.Length);
        if (dot < 0 || !int.TryParse(sourceKey.AsSpan(BlockPrefix.Length, dot - BlockPrefix.Length),
                NumberStyles.None, CultureInfo.InvariantCulture, out int layer) || layer >= _backboneLayers)
            return null;
        return BlockSuffixes.TryGetValue(sourceKey[(dot + 1)..], out string? suffix)
            ? string.Create(CultureInfo.InvariantCulture, $"layers.{layer}.{suffix}")
            : null;
    }
}
