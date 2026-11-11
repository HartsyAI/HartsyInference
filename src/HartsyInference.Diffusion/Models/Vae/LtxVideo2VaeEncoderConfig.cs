namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Config for the LTX-2 causal video VAE encoder (<c>LTX2VideoEncoder3d</c>). Defaults are the SANA-WM
/// <c>vae/config.json</c>. Per-block arrays are in encoder order.</summary>
public sealed record LtxVideo2VaeEncoderConfig
{
    /// <summary>Latent channels: <c>conv_in</c> output width and the mean half of <c>conv_out</c>.</summary>
    public int LatentChannels { get; init; } = 128;

    /// <summary>RGB input channels.</summary>
    public int InChannels { get; init; } = 3;

    /// <summary>Output width of each down block.</summary>
    public int[] BlockOutChannels { get; init; } = [256, 512, 1024, 2048];

    /// <summary>Resnets per down block, then the mid block's count (one entry more than the down blocks).</summary>
    public int[] LayersPerBlock { get; init; } = [4, 6, 6, 2, 2];

    /// <summary>Per down block: whether it carries a downsampler.</summary>
    public bool[] SpatioTemporalScaling { get; init; } = [true, true, true, true];

    /// <summary>Per down block: <c>spatial</c> (1,2,2), <c>temporal</c> (2,1,1) or <c>spatiotemporal</c> (2,2,2).</summary>
    public string[] DownsampleType { get; init; } = ["spatial", "temporal", "spatiotemporal", "spatiotemporal"];

    /// <summary>Spatial patch size of the pixel unshuffle that precedes <c>conv_in</c>.</summary>
    public int PatchSize { get; init; } = 4;

    /// <summary>Temporal patch size. Only 1 is supported by the LTX-2 encoder.</summary>
    public int PatchSizeT { get; init; } = 1;

    /// <summary>Upstream <c>resnet_norm_eps</c>. It only reaches the shortcut LayerNorm, which the encoder never builds
    /// (every encoder resnet keeps its width), so it does not change the output.</summary>
    public float ResnetNormEps { get; init; } = 1e-6f;

    /// <summary>Whether the encoder convolutions are causal (<c>encoder_causal</c>).</summary>
    public bool IsCausal { get; init; } = true;

    /// <summary>Spatial padding of the encoder convolutions (<c>encoder_spatial_padding_mode</c>). Only <c>zeros</c> is implemented.</summary>
    public string SpatialPaddingMode { get; init; } = "zeros";
}
