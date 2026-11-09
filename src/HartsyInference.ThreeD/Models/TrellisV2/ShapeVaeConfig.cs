namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>TRELLIS.2 shape VAE configuration: the encoder and decoder architectures from one scvae config.</summary>
public sealed record ShapeVaeConfig
{
    public required ShapeVaeEncoderConfig Encoder { get; init; }

    public required ShapeVaeDecoderConfig Decoder { get; init; }

    /// <summary>The shipped <c>shape_vae_next_dc_f16c32_fp16</c> architecture (16x downsample, 32 latent channels).</summary>
    public static ShapeVaeConfig NextDcF16C32 => new()
    {
        Encoder = new ShapeVaeEncoderConfig
        {
            ModelChannels = [64, 128, 256, 512, 1024],
            LatentChannels = 32,
            NumBlocks = [0, 4, 8, 16, 4],
            BlockType = ["SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d"],
            DownBlockType = ["SparseResBlockS2C3d", "SparseResBlockS2C3d", "SparseResBlockS2C3d", "SparseResBlockS2C3d"],
            UseCheckpoint = [true, true, false, false, false],
            UseFp16 = true,
        },
        Decoder = new ShapeVaeDecoderConfig
        {
            Resolution = 256,
            ModelChannels = [1024, 512, 256, 128, 64],
            LatentChannels = 32,
            NumBlocks = [4, 16, 8, 4, 0],
            BlockType = ["SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d"],
            UpBlockType = ["SparseResBlockC2S3d", "SparseResBlockC2S3d", "SparseResBlockC2S3d", "SparseResBlockC2S3d"],
            UseCheckpoint = [false, false, false, true, true],
            UseFp16 = true,
        },
    };

    /// <summary>Validates both halves; the encoder and decoder constructors call this on their own config.</summary>
    public void Validate()
    {
        Encoder.Validate();
        Decoder.Validate();
    }
}
