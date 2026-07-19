namespace HartsyInference.Engine.Services;

/// <summary>The single in-process entry point for running inference: owns the compute backend and the loaded-model
/// cache, and exposes one typed service per capability. A consumer (CLI, HTTP API, SwarmUI extension, direct library
/// use) constructs one, calls the service it needs, and disposes it.</summary>
public interface IInferenceEngine : IDisposable
{
    /// <summary>The active backend selector (auto/cpu/cuda/vulkan).</summary>
    string BackendSelector { get; }

    /// <summary>Human-readable description of what the selector resolves to.</summary>
    string BackendDescription { get; }

    /// <summary>Switches the compute backend, disposing every loaded model bound to the old device.</summary>
    void SetBackend(string selector);

    /// <summary>Whether a handler is wired for <paramref name="modality"/>.</summary>
    bool IsSupported(Modality modality);

    /// <summary>Image generation (all diffusion families + composition features).</summary>
    IImagesService Images { get; }

    /// <summary>Video generation.</summary>
    IVideoService Video { get; }

    /// <summary>Chat / text generation, including the multimodal VLM path.</summary>
    ITextService Text { get; }

    /// <summary>Text-to-music generation.</summary>
    IMusicService Music { get; }

    /// <summary>Text-to-speech synthesis.</summary>
    ISpeechService Speech { get; }

    /// <summary>Speech-to-text transcription.</summary>
    ITranscribeService Transcribe { get; }

    /// <summary>Voice conversion.</summary>
    IVoiceConversionService VoiceConversion { get; }

    /// <summary>Audio effects: stem separation and enhancement.</summary>
    IFxService Fx { get; }

    /// <summary>Vision: embed / detect / segment.</summary>
    IVisionService Vision { get; }

    /// <summary>3D mesh generation.</summary>
    IMeshService Mesh { get; }

    /// <summary>Interactive world sessions.</summary>
    IWorldService World { get; }
}
