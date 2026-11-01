namespace HartsyInference.Core.Configuration;

public static partial class EngineKnobs
{
    // One override per top-level model folder. Unset keeps the default name in ModelFolders (the name SwarmUI uses).

    /// <summary>Folder under the models root for LLM models; unset uses <c>llm</c>.</summary>
    public static readonly Knob<string?> ModelFolderLlm =
        Str("paths.folder.llm", null, KnobScope.Runtime, KnobDomain.Paths, "Folder under the models root for LLM models; unset uses llm.");

    /// <summary>Folder under the models root for audio models; unset uses <c>audio</c>.</summary>
    public static readonly Knob<string?> ModelFolderAudio =
        Str("paths.folder.audio", null, KnobScope.Runtime, KnobDomain.Paths, "Folder under the models root for audio models; unset uses audio.");

    /// <summary>Folder under the models root for Stable Diffusion checkpoints; unset uses <c>Stable-Diffusion</c>.</summary>
    public static readonly Knob<string?> ModelFolderStableDiffusion =
        Str("paths.folder.stableDiffusion", null, KnobScope.Runtime, KnobDomain.Paths, "Folder under the models root for Stable Diffusion checkpoints; unset uses Stable-Diffusion.");

    /// <summary>Folder under the models root for VAE weights; unset uses <c>VAE</c>.</summary>
    public static readonly Knob<string?> ModelFolderVae =
        Str("paths.folder.vae", null, KnobScope.Runtime, KnobDomain.Paths, "Folder under the models root for VAE weights; unset uses VAE.");

    /// <summary>Folder under the models root for text encoders; unset uses <c>text_encoders</c>.</summary>
    public static readonly Knob<string?> ModelFolderTextEncoders =
        Str("paths.folder.textEncoders", null, KnobScope.Runtime, KnobDomain.Paths, "Folder under the models root for text encoders; unset uses text_encoders.");

    /// <summary>Folder under the models root for CLIP vision encoders; unset uses <c>clip_vision</c>.</summary>
    public static readonly Knob<string?> ModelFolderClipVision =
        Str("paths.folder.clipVision", null, KnobScope.Runtime, KnobDomain.Paths, "Folder under the models root for CLIP vision encoders; unset uses clip_vision.");

    /// <summary>Folder under the models root for ControlNet models; unset uses <c>controlnet</c>.</summary>
    public static readonly Knob<string?> ModelFolderControlNet =
        Str("paths.folder.controlNet", null, KnobScope.Runtime, KnobDomain.Paths, "Folder under the models root for ControlNet models; unset uses controlnet.");

    /// <summary>Folder under the models root for LoRA adapters; unset uses <c>Lora</c>.</summary>
    public static readonly Knob<string?> ModelFolderLora =
        Str("paths.folder.lora", null, KnobScope.Runtime, KnobDomain.Paths, "Folder under the models root for LoRA adapters; unset uses Lora.");

    /// <summary>Folder under the models root for embeddings; unset uses <c>Embedding</c>.</summary>
    public static readonly Knob<string?> ModelFolderEmbedding =
        Str("paths.folder.embedding", null, KnobScope.Runtime, KnobDomain.Paths, "Folder under the models root for embeddings; unset uses Embedding.");

    /// <summary>Folder under the models root for IP-Adapter weights; unset uses <c>ipadapter</c>.</summary>
    public static readonly Knob<string?> ModelFolderIpAdapter =
        Str("paths.folder.ipAdapter", null, KnobScope.Runtime, KnobDomain.Paths, "Folder under the models root for IP-Adapter weights; unset uses ipadapter.");

    /// <summary>Folder under the models root for vision models; unset uses <c>Vision</c>.</summary>
    public static readonly Knob<string?> ModelFolderVision =
        Str("paths.folder.vision", null, KnobScope.Runtime, KnobDomain.Paths, "Folder under the models root for vision models; unset uses Vision.");

    /// <summary>Folder under the models root for decision models; unset uses <c>Decision</c>.</summary>
    public static readonly Knob<string?> ModelFolderDecision =
        Str("paths.folder.decision", null, KnobScope.Runtime, KnobDomain.Paths, "Folder under the models root for decision models; unset uses Decision.");

    /// <summary>Folder under the models root for video models; unset uses <c>Video</c>.</summary>
    public static readonly Knob<string?> ModelFolderVideo =
        Str("paths.folder.video", null, KnobScope.Runtime, KnobDomain.Paths, "Folder under the models root for video models; unset uses Video.");

    /// <summary>Folder under the models root for diffusion transformers; unset uses <c>diffusion_models</c>.</summary>
    public static readonly Knob<string?> ModelFolderDiffusionModels =
        Str("paths.folder.diffusionModels", null, KnobScope.Runtime, KnobDomain.Paths, "Folder under the models root for diffusion transformers; unset uses diffusion_models.");

    /// <summary>Folder under the models root for Grounding DINO; unset uses <c>grounding-dino</c>.</summary>
    public static readonly Knob<string?> ModelFolderGroundingDino =
        Str("paths.folder.groundingDino", null, KnobScope.Runtime, KnobDomain.Paths, "Folder under the models root for Grounding DINO; unset uses grounding-dino.");

    /// <summary>Folder under the models root for SAM 2; unset uses <c>sam2</c>.</summary>
    public static readonly Knob<string?> ModelFolderSam2 =
        Str("paths.folder.sam2", null, KnobScope.Runtime, KnobDomain.Paths, "Folder under the models root for SAM 2; unset uses sam2.");

    /// <summary>Folder under the models root for RT-DETR; unset uses <c>rtdetr</c>.</summary>
    public static readonly Knob<string?> ModelFolderRtdetr =
        Str("paths.folder.rtdetr", null, KnobScope.Runtime, KnobDomain.Paths, "Folder under the models root for RT-DETR; unset uses rtdetr.");

    /// <summary>Folder under the models root for ClipSeg; unset uses <c>clipseg</c>.</summary>
    public static readonly Knob<string?> ModelFolderClipseg =
        Str("paths.folder.clipseg", null, KnobScope.Runtime, KnobDomain.Paths, "Folder under the models root for ClipSeg; unset uses clipseg.");
}
