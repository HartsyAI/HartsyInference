using HartsyInference.World.Camera;

namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>Loads the SANA-WM components one stage at a time. <see cref="SanaWmPipeline"/> opens them in the order text
/// encoder, image encoder, DiT, video decoder and disposes each stage before opening the next, so only one component's
/// weights are resident at once.</summary>
public interface ISanaWmComponents
{
    /// <summary>Loads the Gemma-2 text encoder and its tokenizer.</summary>
    SanaWmStage<ISanaWmTextEncoder> OpenTextEncoder();

    /// <summary>Loads the LTX-2 VAE encoder.</summary>
    SanaWmStage<ISanaWmImageEncoder> OpenImageEncoder();

    /// <summary>Loads the DiT, bound to the run's conditioning. The stage borrows <paramref name="text"/> and
    /// <paramref name="camera"/>; both outlive it.</summary>
    SanaWmStage<ISanaWmDenoiser> OpenDenoiser(SanaWmTextConditioning text, SanaWmCameraConditioning camera);

    /// <summary>Loads the LTX-2 VAE decoder.</summary>
    SanaWmStage<ISanaWmVideoDecoder> OpenVideoDecoder();
}
