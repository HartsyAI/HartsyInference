namespace HartsyInference.Audio.Models.Auk;

/// <summary>Shape of the AuK BigVGAN-flow VAE (decoder, encoder and latent statistics); defaults are the 24 kHz release with 64-d latents and hop 480.</summary>
public sealed record AukVaeConfig
{
    /// <summary>Decoder transposed-conv strides, one per upsample stage; kernel is 2x stride.</summary>
    public int[] UpsampleRates { get; init; } = [5, 4, 3, 2, 2, 2];

    /// <summary>Decoder width after conv_pre; stage i outputs <c>InitialChannels &gt;&gt; (i + 1)</c>.</summary>
    public int InitialChannels { get; init; } = 1_536;

    /// <summary>AMPBlock kernel sizes; every stage runs one block per kernel and averages them.</summary>
    public int[] ResblockKernelSizes { get; init; } = [3, 7, 11];

    /// <summary>Dilations of the three conv pairs inside each AMPBlock (second convs are dilation 1).</summary>
    public int[] ResblockDilations { get; init; } = [1, 3, 5];

    /// <summary>Latent channels.</summary>
    public int LatentDim { get; init; } = 64;

    /// <summary>Encoder strides, one per down block.</summary>
    public int[] DownsampleRates { get; init; } = [2, 2, 2, 3, 4, 5];

    /// <summary>Encoder widths: the stem width followed by one entry per down block.</summary>
    public int[] DownsampleChannels { get; init; } = [12, 24, 48, 96, 192, 384, 768];

    /// <summary>Residual layers per encoder ResStack (dilation 2^i).</summary>
    public int EncoderStackLayers { get; init; } = 6;

    /// <summary>Causal decoder convs and transposed convs (conv_pre is always symmetric).</summary>
    public bool Causal { get; init; } = true;

    /// <summary>Causal down-pass padding in the anti-aliased activations (the up pass is always non-causal).</summary>
    public bool ActCausal { get; init; } = true;

    /// <summary>Taps of the anti-aliasing Kaiser-sinc filters.</summary>
    public int AntiAliasKernel { get; init; } = 12;

    /// <summary>Output sample rate.</summary>
    public int SampleRate { get; init; } = 24_000;

    /// <summary>Samples per latent frame (product of the encoder strides).</summary>
    public int Hop
    {
        get
        {
            int hop = 1;
            foreach (int r in DownsampleRates) hop *= r;
            return hop;
        }
    }

    /// <summary>Width of the last decoder stage (the final snake and conv_post input).</summary>
    public int FinalChannels => InitialChannels >> UpsampleRates.Length;

    /// <summary>Throws when the strides, channel lists or widths cannot form a consistent VAE.</summary>
    public void Validate()
    {
        int up = 1;
        foreach (int r in UpsampleRates) up *= r;
        if (up != Hop) throw new ArgumentException($"Upsample product {up} must equal the encoder hop {Hop}.");
        if (DownsampleChannels.Length != DownsampleRates.Length + 1)
            throw new ArgumentException("DownsampleChannels needs one more entry than DownsampleRates.");
        if (FinalChannels < 1) throw new ArgumentException("InitialChannels is too small for the number of upsample stages.");
        if (ResblockKernelSizes.Length == 0 || ResblockDilations.Length == 0) throw new ArgumentException("Resblock lists must not be empty.");
    }
}
