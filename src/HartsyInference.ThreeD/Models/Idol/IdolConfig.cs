using System.Text.Json;

namespace HartsyInference.ThreeD.Models.Idol;

/// <summary>The architecture of Idol's denoiser (<c>LatentSeqMMFlowModel</c>): an 8192-token latent and one camera token, jointly
/// attended with the image tokens. The released values are the defaults; <see cref="FromJson"/> reads the checkpoint's
/// <c>config.json</c> (<c>denoiser.args</c>).</summary>
public sealed record IdolConfig
{
    /// <summary>Latent tokens (<c>q_token_length</c>): a scrambled Sobol set, stored as <c>pos_pe</c>.</summary>
    public int Tokens { get; init; } = 8192;

    /// <summary>Latent channels per token (<c>in_channels</c>).</summary>
    public int LatentChannels { get; init; } = 16;

    /// <summary>Camera channels (<c>cam_channels</c>); the camera is one token.</summary>
    public int CameraChannels { get; init; } = 5;

    /// <summary>Model width (<c>model_channels</c>).</summary>
    public int ModelChannels { get; init; } = 1024;

    /// <summary>Width of the DINOv3 tokens (<c>cond_channels</c>).</summary>
    public int CondChannels { get; init; } = 1280;

    /// <summary>Width of the FLUX.2 VAE tokens (<c>cond2_channels</c>).</summary>
    public int Cond2Channels { get; init; } = 128;

    /// <summary>Joint blocks over latent, image and camera (<c>num_blocks</c>).</summary>
    public int Blocks { get; init; } = 24;

    /// <summary>Blocks that refine the latent and the image tokens on their own before the joint blocks (<c>num_refiner_blocks</c>).</summary>
    public int RefinerBlocks { get; init; } = 2;

    /// <summary>Attention heads (<c>num_heads</c>).</summary>
    public int Heads { get; init; } = 16;

    /// <summary>Width of one head (<c>num_head_channels</c>).</summary>
    public int HeadDim { get; init; } = 64;

    /// <summary>MLP width over model width (<c>mlp_ratio</c>).</summary>
    public int MlpRatio { get; init; } = 4;

    /// <summary>MLP hidden width.</summary>
    public int MlpHidden => ModelChannels * MlpRatio;

    /// <summary>Tokens of the image condition: the class token, four registers and a 64 × 64 patch grid at 1024 px. Informational: it is in no <c>config.json</c> and the model reads the length of the features it is given.</summary>
    public int CondTokens { get; init; } = 4101;

    /// <summary>The configuration in a checkpoint's <c>config.json</c> (its <c>denoiser.args</c> object); unnamed fields keep their released values.</summary>
    public static IdolConfig FromJson(string json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        if (!doc.RootElement.TryGetProperty("denoiser", out JsonElement denoiser) || !denoiser.TryGetProperty("args", out JsonElement args))
            throw new FormatException("The Idol config has no denoiser.args object.");
        IdolConfig c = new();
        int Read(string name, int fallback) => args.TryGetProperty(name, out JsonElement v) ? (int)v.GetDouble() : fallback;
        c = c with
        {
            Tokens = Read("q_token_length", c.Tokens),
            LatentChannels = Read("in_channels", c.LatentChannels),
            CameraChannels = Read("cam_channels", c.CameraChannels),
            ModelChannels = Read("model_channels", c.ModelChannels),
            CondChannels = Read("cond_channels", c.CondChannels),
            Cond2Channels = Read("cond2_channels", c.Cond2Channels),
            Blocks = Read("num_blocks", c.Blocks),
            RefinerBlocks = Read("num_refiner_blocks", c.RefinerBlocks),
            Heads = Read("num_heads", c.Heads),
            HeadDim = Read("num_head_channels", c.HeadDim),
            MlpRatio = Read("mlp_ratio", c.MlpRatio),
        };
        if (c.ModelChannels != c.Heads * c.HeadDim)
            throw new FormatException($"Idol width {c.ModelChannels} is not {c.Heads} heads of {c.HeadDim}.");
        if (c.HeadDim % 2 != 0)
            throw new FormatException($"Idol's rotary position rotates channel pairs, so a head of {c.HeadDim} channels must be even.");
        return c;
    }
}
