namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Shape of the ControlFoley mel VAE decoder (<c>lib.autoencoder.vae.VAE</c> / <c>Decoder1D</c>).</summary>
public sealed record ControlFoleyVaeConfig
{
    /// <summary>Mel bins produced by the decoder.</summary>
    public required int DataDim { get; init; }

    /// <summary>Latent channels per frame.</summary>
    public required int EmbedDim { get; init; }

    /// <summary>Base channel count; level <c>i</c> uses <c>HiddenDim * ChMult[i]</c>.</summary>
    public required int HiddenDim { get; init; }

    public int[] ChMult { get; init; } = [1, 2, 4];

    public int NumResBlocks { get; init; } = 2;

    public int[] AttnLayers { get; init; } = [3];

    /// <summary>Encoder levels that halve the length; the decoder doubles it after level <c>i + 1</c>.</summary>
    public int[] DownLayers { get; init; } = [0];

    public float ClipAct { get; init; } = 256f;

    /// <summary>The released 44.1 kHz VAE (<c>VAE_44k</c>): 128 mel bins, 40 latent channels.</summary>
    public static ControlFoleyVaeConfig V44k => new() { DataDim = 128, EmbedDim = 40, HiddenDim = 512 };
}
