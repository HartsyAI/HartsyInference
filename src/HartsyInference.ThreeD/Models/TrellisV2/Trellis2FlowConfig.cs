using System.Text.Json;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>TRELLIS.2 flow-transformer architecture, read from a checkpoint's <c>.json</c> sidecar (<c>args</c> of
/// <c>SparseStructureFlowModel</c> or <c>SLatFlowModel</c>). Only the shipped parameterization is implemented: RoPE, shared
/// adaLN modulation, and QK RMS norm on self- and cross-attention.</summary>
public sealed record Trellis2FlowConfig
{
    /// <summary>Frequency-embedding width of the timestep embedder (upstream default).</summary>
    public const int TimestepFrequencyDim = 256;

    /// <summary>Upstream <c>SparseStructureFlowModel</c> class name: a dense grid of tokens.</summary>
    public const string DenseModelName = "SparseStructureFlowModel";

    /// <summary>Upstream <c>SLatFlowModel</c> class name: one token per sparse voxel.</summary>
    public const string SparseModelName = "SLatFlowModel";

    /// <summary>True for the dense sparse-structure flow, false for the sparse structured-latent flow.</summary>
    public required bool Dense { get; init; }

    /// <summary>Grid edge the flow was trained on (<c>resolution</c>); the dense flow takes exactly this grid.</summary>
    public required int Resolution { get; init; }

    public required int InChannels { get; init; }

    public required int OutChannels { get; init; }

    /// <summary>Transformer width (<c>model_channels</c>).</summary>
    public required int ModelChannels { get; init; }

    /// <summary>Channels of the conditioning tokens the cross-attention reads (<c>cond_channels</c>).</summary>
    public required int CondChannels { get; init; }

    public required int NumBlocks { get; init; }

    public required int NumHeads { get; init; }

    /// <summary>MLP expansion (<c>mlp_ratio</c>); the hidden width is <c>int(ModelChannels * MlpRatio)</c>.</summary>
    public required double MlpRatio { get; init; }

    /// <summary>Upstream <c>pe_mode</c>; only <c>rope</c> is implemented.</summary>
    public string PositionEncoding { get; init; } = "rope";

    /// <summary>Upstream <c>share_mod</c>; only true is implemented.</summary>
    public bool ShareModulation { get; init; } = true;

    /// <summary>Upstream <c>qk_rms_norm</c>: per-head RMS norm of the self-attention queries and keys.</summary>
    public bool QkRmsNorm { get; init; } = true;

    /// <summary>Upstream <c>qk_rms_norm_cross</c>: per-head RMS norm of the cross-attention queries and keys.</summary>
    public bool QkRmsNormCross { get; init; } = true;

    /// <summary>Channels per attention head.</summary>
    public int HeadDim => ModelChannels / NumHeads;

    /// <summary>Width of the feed-forward hidden layer; truncates like upstream's <c>int(channels * mlp_ratio)</c>.</summary>
    public int MlpHidden => (int)(ModelChannels * MlpRatio);

    /// <summary>Tokens of a dense flow's grid: <c>Resolution³</c>.</summary>
    public int DenseTokens => Resolution * Resolution * Resolution;

    /// <summary>Throws <see cref="ArgumentException"/> or <see cref="NotSupportedException"/> when the architecture cannot be realized.</summary>
    public void Validate()
    {
        if (Resolution <= 0 || InChannels <= 0 || OutChannels <= 0 || ModelChannels <= 0 || CondChannels <= 0 || NumBlocks <= 0 || NumHeads <= 0)
            throw new ArgumentException("Flow config dimensions must all be positive.");
        if (ModelChannels % NumHeads != 0)
            throw new ArgumentException($"Flow width {ModelChannels} does not divide over {NumHeads} heads.");
        if (HeadDim % 2 != 0 || HeadDim < 6)
            throw new ArgumentException($"Flow head width {HeadDim} must be even and at least 6 to carry a 3-axis RoPE.");
        if (MlpHidden <= 0)
            throw new ArgumentException($"MLP ratio {MlpRatio} gives no hidden width.");
        if (PositionEncoding != "rope")
            throw new NotSupportedException($"Flow position encoding '{PositionEncoding}' is not implemented; only rope is.");
        if (!ShareModulation)
            throw new NotSupportedException("Flows without shared modulation (share_mod=false) are not implemented.");
    }

    /// <summary>Parses the checkpoint sidecar JSON (<c>{"name": ..., "args": {...}}</c>) and validates it.</summary>
    public static Trellis2FlowConfig FromJson(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using JsonDocument doc = JsonDocument.Parse(json);
        JsonElement root = doc.RootElement;
        string name = root.GetProperty("name").GetString() ?? string.Empty;
        bool dense = name switch
        {
            DenseModelName => true,
            SparseModelName => false,
            _ => throw new NotSupportedException($"Flow model '{name}' is not a TRELLIS.2 flow; expected {DenseModelName} or {SparseModelName}."),
        };
        JsonElement args = root.GetProperty("args");
        if (args.TryGetProperty("rope_freq", out JsonElement freq) && freq.ValueKind == JsonValueKind.Array
            && (freq[0].GetDouble() != 1.0 || freq[1].GetDouble() != 10000.0))
            throw new NotSupportedException("Flows with a rope_freq other than (1, 10000) are not implemented.");
        int numHeads = args.TryGetProperty("num_heads", out JsonElement heads) && heads.ValueKind == JsonValueKind.Number
            ? heads.GetInt32()
            : args.GetProperty("model_channels").GetInt32() / (args.TryGetProperty("num_head_channels", out JsonElement hc) ? hc.GetInt32() : 64);
        Trellis2FlowConfig config = new()
        {
            Dense = dense,
            Resolution = args.GetProperty("resolution").GetInt32(),
            InChannels = args.GetProperty("in_channels").GetInt32(),
            OutChannels = args.GetProperty("out_channels").GetInt32(),
            ModelChannels = args.GetProperty("model_channels").GetInt32(),
            CondChannels = args.GetProperty("cond_channels").GetInt32(),
            NumBlocks = args.GetProperty("num_blocks").GetInt32(),
            NumHeads = numHeads,
            MlpRatio = args.TryGetProperty("mlp_ratio", out JsonElement ratio) ? ratio.GetDouble() : 4.0,
            PositionEncoding = args.TryGetProperty("pe_mode", out JsonElement pe) ? pe.GetString() ?? "ape" : "ape",
            ShareModulation = args.TryGetProperty("share_mod", out JsonElement share) && share.GetBoolean(),
            QkRmsNorm = args.TryGetProperty("qk_rms_norm", out JsonElement qk) && qk.GetBoolean(),
            QkRmsNormCross = args.TryGetProperty("qk_rms_norm_cross", out JsonElement qkc) && qkc.GetBoolean(),
        };
        config.Validate();
        return config;
    }
}
