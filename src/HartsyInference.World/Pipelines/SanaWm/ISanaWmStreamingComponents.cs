using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Streaming;

namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>Loads the components of a SANA-WM stream. The text encoder and the seed-frame encoder are opened and released
/// one after the other before generation starts; the chunk denoiser and the streaming decoder are then opened together and
/// stay resident for the whole stream, since every chunk needs both.</summary>
public interface ISanaWmStreamingComponents
{
    /// <summary>Loads the Gemma-2 text encoder and its tokenizer.</summary>
    SanaWmStage<ISanaWmTextEncoder> OpenTextEncoder();

    /// <summary>Loads the causal LTX-2 VAE's encoder.</summary>
    SanaWmStage<ISanaWmImageEncoder> OpenImageEncoder();

    /// <summary>Loads the streaming DiT, bound to the stream's caption. The stage borrows <paramref name="text"/>, which
    /// outlives it.</summary>
    /// <param name="softmaxWindow">How many earlier chunks the softmax blocks attend to.</param>
    SanaWmStage<ISanaWmChunkDenoiser> OpenChunkDenoiser(SanaWmTextConditioning text, TokenWindowPolicy softmaxWindow);

    /// <summary>Loads the causal LTX-2 VAE's decoder as a streaming decoder.</summary>
    SanaWmStage<IStreamingVideoDecoder> OpenStreamingDecoder();
}
