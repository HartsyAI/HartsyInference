namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>TRELLIS.2 shape VAE encoder architecture; mirrors <c>models.encoder.args</c> of the upstream scvae config.</summary>
public sealed record ShapeVaeEncoderConfig
{
    /// <summary>Channels per level, finest first (<c>model_channels</c>).</summary>
    public required int[] ModelChannels { get; init; }

    /// <summary>Latent channels (<c>latent_channels</c>); the encoder returns the posterior mean with this many channels.</summary>
    public required int LatentChannels { get; init; }

    /// <summary>Residual blocks per level, before that level's downsample (<c>num_blocks</c>).</summary>
    public required int[] NumBlocks { get; init; }

    /// <summary>Block type per level (<c>block_type</c>); only <c>SparseConvNeXtBlock3d</c> is implemented.</summary>
    public required string[] BlockType { get; init; }

    /// <summary>Downsample type per transition (<c>down_block_type</c>); only <c>SparseResBlockS2C3d</c> is implemented.</summary>
    public required string[] DownBlockType { get; init; }

    /// <summary>Per-level <c>block_args.use_checkpoint</c>. Activation checkpointing is a training concern, so inference ignores it.</summary>
    public required bool[] UseCheckpoint { get; init; }

    /// <summary>Upstream <c>use_fp16</c>. The CPU reference always computes in F32.</summary>
    public required bool UseFp16 { get; init; }

    /// <summary>Throws <see cref="ArgumentException"/> when the architecture cannot be realized by the blocks.</summary>
    public void Validate()
    {
        ShapeVaeConfigRules.ValidateLevels("encoder", ModelChannels, NumBlocks, BlockType, UseCheckpoint);
        ShapeVaeConfigRules.ValidateTransitionTypes("encoder", DownBlockType, ModelChannels.Length - 1, ShapeVaeConfigRules.SpatialToChannelBlock);
        if (LatentChannels <= 0) throw new ArgumentException($"encoder latent channels must be positive, got {LatentChannels}.");
        for (int i = 0; i + 1 < ModelChannels.Length; i++)
        {
            int cin = ModelChannels[i], cout = ModelChannels[i + 1];
            // Space-to-channel: conv1 emits cout/8 channels per voxel, and the 8·cin skip splits evenly into cout groups.
            if (cout % ShapeVaeConfigRules.ChildGroup != 0 || ShapeVaeConfigRules.ChildGroup * cin % cout != 0)
                throw new ArgumentException(
                    $"encoder level {i} to {i + 1}: {cin} -> {cout} needs cout divisible by 8 and 8*cin divisible by cout.");
        }
    }
}
