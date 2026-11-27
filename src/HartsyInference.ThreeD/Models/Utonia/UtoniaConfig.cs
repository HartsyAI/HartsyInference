using HartsyInference.Core.Exceptions;

namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>Point Transformer V3 hyper-parameters of a Utonia checkpoint, named after the PTv3 constructor arguments.</summary>
/// <remarks>Read these from the checkpoint's <c>config</c> entry. CoLa3D overrides enc_mode, traceable and shuffle_orders.</remarks>
public sealed record UtoniaConfig
{
    /// <summary>Per-point input width. CoLa3D feeds 9: centred xyz, color, normal.</summary>
    public required int InChannels { get; init; }

    /// <summary>Serialization curves in checkpoint names (<c>z</c>, <c>z-trans</c>, <c>hilbert</c>, <c>hilbert-trans</c>). Block i uses curve i % length.</summary>
    public required string[] Order { get; init; }

    /// <summary>Grid stride of each pooling step; its length is the stage count minus one.</summary>
    public required int[] Stride { get; init; }

    public required int[] EncDepths { get; init; }
    public required int[] EncChannels { get; init; }
    public required int[] EncNumHeads { get; init; }
    public required int[] EncPatchSize { get; init; }

    /// <summary>When true, no decoder is built. Encoder-only checkpoints such as CoLa3D's use this.</summary>
    public required bool EncoderOnly { get; init; }

    public int[] DecDepths { get; init; } = [];
    public int[] DecChannels { get; init; } = [];
    public int[] DecNumHeads { get; init; } = [];
    public int[] DecPatchSize { get; init; } = [];
    public double MlpRatio { get; init; } = 4.0;
    public bool QkvBias { get; init; } = true;
    public bool PreNorm { get; init; } = true;

    /// <summary>LayerScale initial value; null means no LayerScale parameters exist in the checkpoint.</summary>
    public float? LayerScaleInit { get; init; }

    public float RopeBase { get; init; } = 10f;

    /// <summary>Whether attention rotates queries and keys with the 3D RoPE of Utonia. Sonata (the P3-SAM backbone) has none: its attention is plain scaled
    /// dot product over the serialised patch, and its checkpoint carries no <c>attn.rope.inv_freq</c> buffers.</summary>
    public bool UseRope { get; init; } = true;

    /// <summary>Checks the shape and divisibility rules the modules rely on. Throws for an unusable config.</summary>
    /// <summary>
    /// Encoder of the CoLa3D Utonia checkpoint. Depths, widths, the 18-channel head size (from the RoPE buffer's
    /// length) and the 2x stride per stage are read from <c>utonia.pth</c>. Curve order, patch size and RoPE base are
    /// not stored in the weights and follow Pointcept's defaults, so they are unverified against the reference.
    /// </summary>
    public static UtoniaConfig Cola3DEncoder => new()
    {
        InChannels = 9,
        Order = ["z", "z-trans", "hilbert", "hilbert-trans"],
        Stride = [2, 2, 2, 2],
        EncDepths = [3, 3, 3, 12, 3],
        EncChannels = [54, 108, 216, 432, 576],
        EncNumHeads = [3, 6, 12, 24, 32],
        EncPatchSize = [1024, 1024, 1024, 1024, 1024],
        EncoderOnly = true,
        MlpRatio = 4.0,
        QkvBias = true,
        PreNorm = true,
        LayerScaleInit = null,
    };

    /// <summary>
    /// Sonata (<c>facebook/sonata</c>, the Point Transformer V3 backbone of Hunyuan3D-Part's P3-SAM; its weights sit inside <c>p3sam.safetensors</c> under
    /// <c>sonata.</c>): five stages of 48, 96, 192, 384 and 512 channels with 3, 3, 3, 12 and 3 blocks, 3 to 32 heads, patch 1024, the four curves,
    /// stride 2, a CPE per block, no RoPE and no decoder. The widths, depths and head counts are read from the checkpoint header and Hunyuan3D-Part's
    /// <c>sonata.json</c>; the stride and curve order follow that file too.
    /// </summary>
    public static UtoniaConfig Sonata => new()
    {
        InChannels = 9,
        Order = ["z", "z-trans", "hilbert", "hilbert-trans"],
        Stride = [2, 2, 2, 2],
        EncDepths = [3, 3, 3, 12, 3],
        EncChannels = [48, 96, 192, 384, 512],
        EncNumHeads = [3, 6, 12, 24, 32],
        EncPatchSize = [1024, 1024, 1024, 1024, 1024],
        EncoderOnly = true,
        MlpRatio = 4.0,
        QkvBias = true,
        PreNorm = true,
        LayerScaleInit = null,
        UseRope = false,
    };

    public void Validate()
    {
        if (InChannels <= 0)
        {
            throw Invalid("InChannels must be positive");
        }

        if (Order is null || Order.Length == 0)
        {
            throw Invalid("Order must name at least one curve");
        }

        foreach (string name in Order)
        {
            _ = UtoniaSerialization.ParseOrder(name);
        }

        int stages = EncDepths?.Length ?? 0;
        if (stages == 0)
        {
            throw Invalid("at least one encoder stage is required");
        }

        ValidateStages("enc", EncDepths!, EncChannels, EncNumHeads, EncPatchSize, stages, UseRope);
        if (Stride is null || Stride.Length != stages - 1)
        {
            throw Invalid($"Stride must have {stages - 1} entries for {stages} encoder stages");
        }

        foreach (int stride in Stride)
        {
            if (stride <= 0)
            {
                throw Invalid("Stride entries must be positive");
            }
        }

        if (!EncoderOnly)
        {
            ValidateStages("dec", DecDepths, DecChannels, DecNumHeads, DecPatchSize, stages - 1, UseRope);
        }

        if (MlpRatio <= 0)
        {
            throw Invalid("MlpRatio must be positive");
        }

        if (RopeBase <= 0)
        {
            throw Invalid("RopeBase must be positive");
        }
    }

    private static void ValidateStages(string name, int[] depths, int[] channels, int[] heads, int[] patch, int count, bool rope)
    {
        if (depths.Length != count || channels.Length != count || heads.Length != count || patch.Length != count)
        {
            throw Invalid($"{name} arrays must each have {count} entries");
        }

        for (int s = 0; s < count; s++)
        {
            if (depths[s] < 0 || channels[s] <= 0 || heads[s] <= 0 || patch[s] <= 0)
            {
                throw Invalid($"{name} stage {s} needs depth >= 0 and positive channels, heads and patch size");
            }

            if (channels[s] % heads[s] != 0)
            {
                throw Invalid($"{name} stage {s}: channels {channels[s]} not divisible by heads {heads[s]}");
            }

            // 3D RoPE splits each head into x, y, z chunks, and each chunk into rotary pairs.
            if (rope && depths[s] > 0 && (channels[s] / heads[s]) % 6 != 0)
            {
                throw Invalid($"{name} stage {s}: head dim {channels[s] / heads[s]} must be divisible by 6 for 3D RoPE");
            }
        }
    }

    private static HartsyInferenceException Invalid(string message) => new($"Utonia config: {message}.");
}
