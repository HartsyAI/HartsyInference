using System.Text.Json;

namespace HartsyInference.Audio.Models.Haru;

/// <summary>The <c>head</c> block of Haru's <c>config.json</c>: the MeanFlow head that turns one language-model hidden state into
/// the next audio latent.</summary>
internal sealed record HaruHeadConfig
{
    public required int HiddenSize { get; init; }

    public required int LatentDim { get; init; }

    public required int NumLayers { get; init; }

    public required float FfnRatio { get; init; }

    public required float NormEps { get; init; }

    /// <summary>The interval input <c>dt</c> the network is conditioned on, a constant of the training recipe.</summary>
    public required float IntervalEmbed { get; init; }

    /// <summary>Sampling steps; the released head is distilled for one.</summary>
    public required int NumSteps { get; init; }

    public required float GuidanceScale { get; init; }

    /// <summary>The feed-forward width, <c>HiddenSize * FfnRatio</c>.</summary>
    public int FfnSize => (int)(HiddenSize * FfnRatio);

    internal static HaruHeadConfig FromJson(JsonElement head)
    {
        HaruHeadConfig config = new()
        {
            HiddenSize = head.GetProperty("hidden_size").GetInt32(),
            LatentDim = head.GetProperty("latent_dim").GetInt32(),
            NumLayers = head.GetProperty("num_layers").GetInt32(),
            FfnRatio = (float)head.GetProperty("ffn_ratio").GetDouble(),
            NormEps = (float)head.GetProperty("norm_eps").GetDouble(),
            IntervalEmbed = (float)head.GetProperty("interval_embed").GetDouble(),
            NumSteps = head.GetProperty("num_steps").GetInt32(),
            GuidanceScale = (float)head.GetProperty("guidance_scale").GetDouble(),
        };
        if (config.NumLayers < 1 || config.NumSteps < 1 || config.GuidanceScale <= 0 || config.FfnSize < 1)
        {
            throw new InvalidDataException("The Haru head needs at least one layer and one step, a positive guidance scale and a feed-forward width.");
        }
        return config;
    }
}
