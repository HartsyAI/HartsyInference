using SharpInference.Core.Tensors;

namespace SharpInference.ModelHandler.Gguf;

/// <summary>Per-tensor quantization mix policies. Mirrors llama.cpp's <c>LLAMA_FTYPE_*</c> mix policies (Q4_K_S / Q4_K_M / Q5_K_M / Q8_0). The policy decides which DType each tensor gets quantized to based on the tensor's name and shape — small tensors stay at F16 (norms, biases, embeddings), large linear weights take the bulk quant, and a select set of "high-priority" weights get a higher-fidelity quant.</summary>
public sealed record GgufQuantPolicy
{
    /// <summary>Default DType for large 2D+ weights (the backbone of the model).</summary>
    public required DType BackboneDType { get; init; }

    /// <summary>Higher-fidelity DType used for "high-priority" weights — typically output projections and attention V projections, where quality loss is most visible. <c>null</c> means use the backbone DType for everything.</summary>
    public DType? HighFidelityDType { get; init; }

    /// <summary>Predicate that identifies "high-priority" tensors by name. Returns true for tensors that should use <see cref="HighFidelityDType"/> instead of <see cref="BackboneDType"/>.</summary>
    public Func<string, bool>? IsHighFidelity { get; init; }

    /// <summary>Predicate that identifies tensors which should remain at F16 instead of being quantized. Defaults to: 1D tensors (norms / biases) + token embeddings + any weight with fewer elements than <see cref="MinElementsToQuantize"/>.</summary>
    public Func<string, Tensor, bool>? ShouldKeepF16 { get; init; }

    /// <summary>Minimum element count for quantization. Tensors smaller than this stay at F16 — quantization overhead (block headers) outweighs the storage savings on tiny tensors. Default 256 (one super-block).</summary>
    public int MinElementsToQuantize { get; init; } = 256;

    /// <summary>Decides the target DType for one tensor. Routing: <c>ShouldKeepF16</c> wins (returns F16) → <c>IsHighFidelity</c> picks <see cref="HighFidelityDType"/> → otherwise <see cref="BackboneDType"/>.</summary>
    public DType ResolveTargetDType(string tensorName, Tensor tensor)
    {
        if (tensor.Shape.Rank <= 1) return DType.F16;
        if (tensor.Shape.ElementCount < MinElementsToQuantize) return DType.F16;
        if (ShouldKeepF16 is not null && ShouldKeepF16(tensorName, tensor)) return DType.F16;
        if (IsHighFidelity is not null && IsHighFidelity(tensorName) && HighFidelityDType is not null)
            return HighFidelityDType.Value;
        return BackboneDType;
    }

    /// <summary>Q8_0 — uniform 8-bit quant on the backbone, F16 for norms/biases/small. ~50% size of F16, very low quality loss. The conservative default.</summary>
    public static GgufQuantPolicy Q8_0 => new()
    {
        BackboneDType = DType.Q8_0,
        HighFidelityDType = null,
        ShouldKeepF16 = DefaultKeepF16,
    };

    /// <summary>Q4_K_S ("small") — uniform Q4_K on the backbone, F16 for norms. ~25% size of F16. Cheapest 4-bit; lower quality than Q4_K_M.</summary>
    public static GgufQuantPolicy Q4_K_S => new()
    {
        BackboneDType = DType.Q4_K,
        HighFidelityDType = null,
        ShouldKeepF16 = DefaultKeepF16,
    };

    /// <summary>Q4_K_M ("medium") — Q4_K backbone + Q6_K for output projections + attention V. Mirrors llama.cpp's most-popular mix; ~30% size of F16, much better quality than Q4_K_S.</summary>
    public static GgufQuantPolicy Q4_K_M => new()
    {
        BackboneDType = DType.Q4_K,
        HighFidelityDType = DType.Q6_K,
        IsHighFidelity = IsAttentionVOrOutputProj,
        ShouldKeepF16 = DefaultKeepF16,
    };

    /// <summary>Q5_K_M ("medium") — Q5_K backbone + Q6_K for output projections + attention V. ~37% size of F16, excellent quality preservation.</summary>
    public static GgufQuantPolicy Q5_K_M => new()
    {
        BackboneDType = DType.Q5_K,
        HighFidelityDType = DType.Q6_K,
        IsHighFidelity = IsAttentionVOrOutputProj,
        ShouldKeepF16 = DefaultKeepF16,
    };

    /// <summary>Q6_K — uniform Q6_K on the backbone. ~44% size of F16, near-lossless. The "I want quant savings without compromising quality" preset.</summary>
    public static GgufQuantPolicy Q6_K => new()
    {
        BackboneDType = DType.Q6_K,
        HighFidelityDType = null,
        ShouldKeepF16 = DefaultKeepF16,
    };

    private static bool DefaultKeepF16(string name, Tensor t)
    {
        string lower = name.ToLowerInvariant();
        if (lower.Contains(".norm", StringComparison.Ordinal)) return true;
        if (lower.Contains("layernorm", StringComparison.Ordinal)) return true;
        if (lower.Contains("rmsnorm", StringComparison.Ordinal)) return true;
        if (lower.EndsWith(".bias", StringComparison.Ordinal)) return true;
        if (lower.Contains("embed_tokens", StringComparison.Ordinal)) return true;
        if (lower.Contains("token_embd", StringComparison.Ordinal)) return true;
        if (lower.Contains("position_embd", StringComparison.Ordinal)) return true;
        if (lower.Contains("pos_embed", StringComparison.Ordinal)) return true;
        if (lower.Contains("register_tokens", StringComparison.Ordinal)) return true;
        return false;
    }

    /// <summary>Default high-fidelity predicate matching llama.cpp's `_M` mix policy: V-projection (`attn_v`, `to_v`, `v_proj`) and output projection (`output.weight`, `final_proj`, `lm_head`) get the bumped quant.</summary>
    private static bool IsAttentionVOrOutputProj(string name)
    {
        string lower = name.ToLowerInvariant();
        if (lower.Contains(".attn_v.", StringComparison.Ordinal)) return true;
        if (lower.Contains(".to_v.", StringComparison.Ordinal)) return true;
        if (lower.Contains(".v_proj.", StringComparison.Ordinal)) return true;
        if (lower.EndsWith("/output.weight", StringComparison.Ordinal) || lower.Equals("output.weight", StringComparison.Ordinal)) return true;
        if (lower.Contains("final_proj.weight", StringComparison.Ordinal)) return true;
        if (lower.Contains("lm_head.weight", StringComparison.Ordinal)) return true;
        if (lower.Contains("proj_out.weight", StringComparison.Ordinal)) return true;
        return false;
    }
}
