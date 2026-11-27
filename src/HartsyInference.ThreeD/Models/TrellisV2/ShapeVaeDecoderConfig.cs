namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>TRELLIS.2 shape VAE decoder architecture; mirrors <c>models.decoder.args</c> of the upstream scvae config.</summary>
public sealed record ShapeVaeDecoderConfig
{
    /// <summary>Dual-grid resolution the field is meshed at (<c>resolution</c>); used by mesh extraction, not the network.</summary>
    public required int Resolution { get; init; }

    /// <summary>Channels per level, coarsest first (<c>model_channels</c>).</summary>
    public required int[] ModelChannels { get; init; }

    /// <summary>Latent channels (<c>latent_channels</c>) the decoder consumes.</summary>
    public required int LatentChannels { get; init; }

    /// <summary>Residual blocks per level, before that level's upsample (<c>num_blocks</c>).</summary>
    public required int[] NumBlocks { get; init; }

    /// <summary>Block type per level (<c>block_type</c>); only <c>SparseConvNeXtBlock3d</c> is implemented.</summary>
    public required string[] BlockType { get; init; }

    /// <summary>Upsample type per transition (<c>up_block_type</c>); only <c>SparseResBlockC2S3d</c> is implemented.</summary>
    public required string[] UpBlockType { get; init; }

    /// <summary>Per-level <c>block_args.use_checkpoint</c>. Activation checkpointing is a training concern, so inference ignores it.</summary>
    public required bool[] UseCheckpoint { get; init; }

    /// <summary>Upstream <c>use_fp16</c>. The CPU reference always computes in F32.</summary>
    public required bool UseFp16 { get; init; }

    /// <summary>Head width (<c>out_channels</c>): 7 for the shape decoder, 6 for the texture decoder.</summary>
    public int OutputChannels { get; init; } = ShapeVaeKeyMap.DecoderOutputChannels;

    /// <summary>Upstream <c>pred_subdiv</c>. False drops each upsample's <c>to_subdiv</c> head: the children come from another
    /// decoder's subdivisions (<c>guide_subs</c>), as in the texture decoder.</summary>
    public bool PredictSubdivision { get; init; } = true;

    /// <summary>The shipped <c>tex_dec_next_dc_f16c32_fp16</c> architecture: the shape decoder's body with a 6-channel PBR head and
    /// no subdivision heads.</summary>
    public static ShapeVaeDecoderConfig Texture => ShapeVaeConfig.NextDcF16C32.Decoder with
    {
        OutputChannels = ShapeVaeKeyMap.TextureOutputChannels,
        PredictSubdivision = false,
    };

    /// <summary>Throws <see cref="ArgumentException"/> when the architecture cannot be realized by the blocks.</summary>
    public void Validate()
    {
        if (OutputChannels <= 0) throw new ArgumentException($"decoder output channels must be positive, got {OutputChannels}.");
        ShapeVaeConfigRules.ValidateLevels("decoder", ModelChannels, NumBlocks, BlockType, UseCheckpoint);
        ShapeVaeConfigRules.ValidateTransitionTypes("decoder", UpBlockType, ModelChannels.Length - 1, ShapeVaeConfigRules.ChannelToSpatialBlock);
        if (LatentChannels <= 0) throw new ArgumentException($"decoder latent channels must be positive, got {LatentChannels}.");
        if (Resolution <= 0) throw new ArgumentException($"decoder resolution must be positive, got {Resolution}.");
        for (int i = 0; i + 1 < ModelChannels.Length; i++)
        {
            int cin = ModelChannels[i], cout = ModelChannels[i + 1];
            // Channel-to-spatial: the 8 children share cin/8 channels each, so cout must split evenly over them.
            if (cin % ShapeVaeConfigRules.ChildGroup != 0 || cout % (cin / ShapeVaeConfigRules.ChildGroup) != 0)
                throw new ArgumentException(
                    $"decoder level {i} to {i + 1}: {cin} -> {cout} needs cin divisible by 8 and cout divisible by cin/8.");
        }
    }
}
