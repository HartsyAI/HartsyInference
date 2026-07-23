namespace HartsyInference.Engine;

/// <summary>The inference domains the CLI can drive, one per top-level command family.</summary>
public enum Modality
{
    /// <summary>Text-to-image diffusion.</summary>
    Image,

    /// <summary>LLM text generation.</summary>
    Text,

    /// <summary>Text-to-speech synthesis.</summary>
    Speech,

    /// <summary>Music / general audio generation.</summary>
    Music,

    /// <summary>Speech-to-text transcription.</summary>
    Transcribe,

    /// <summary>Vision embeddings, detection, segmentation, and face tasks.</summary>
    Vision,

    /// <summary>Text/image-to-video generation.</summary>
    Video,

    /// <summary>Image/text-to-3D mesh generation.</summary>
    Mesh,

    /// <summary>Action-conditioned interactive world models.</summary>
    World,

    /// <summary>Voice conversion / re-voicing (RVC, OpenVoice tone-color transfer).</summary>
    VoiceConvert,

    /// <summary>Audio effects: stem separation (Demucs), speech enhancement (Resemble-Enhance).</summary>
    Fx,

    /// <summary>Text-to-vector embeddings (RAG/semantic-search style dense sentence vectors) — not yet
    /// CLI-drivable, only reachable via the HTTP API's native/OpenAI-compat routes.</summary>
    Embedding,
}
