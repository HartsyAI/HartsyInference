using HartsyInference.Core.Backends;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Multimodal;
using HartsyInference.LLM.Ssm;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Engine.Services;

/// <summary>One compute device's loaded LLM state owned by <see cref="TextService"/>. A single model/backend/pipeline
/// is not safe for concurrent use, so each slot carries its own lock — but two different slots (e.g. cuda:0 + cpu) can
/// generate at the same time. Exactly one of <see cref="Model"/> / <see cref="SsmModel"/> is set once loaded.</summary>
internal sealed class TextDeviceSlot
{
    /// <summary>Serializes generation on this slot; different slots run concurrently.</summary>
    public SemaphoreSlim Lock { get; } = new SemaphoreSlim(1, 1);

    /// <summary>The compute backend (CPU/CUDA) bound to this slot's device. For a layer-split load this is the
    /// LAST stage's backend (logits/sampling live there).</summary>
    public IBackend? Backend { get; set; }

    /// <summary>The other stage backends of a layer-split load (everything except <see cref="Backend"/>).
    /// Slot-owned: disposed with the slot, exactly like <see cref="Backend"/>.</summary>
    public List<IBackend>? ExtraStageBackends { get; set; }

    /// <summary>The active layer-split plan, or null for a single-device load.</summary>
    public LlmPlacement? Placement { get; set; }

    /// <summary>The loaded GGUF transformer model, or null when nothing (or an SSM model) is loaded.</summary>
    public GgufLanguageModel? Model { get; set; }

    /// <summary>The generation pipeline built for <see cref="Model"/> on <see cref="Backend"/>.</summary>
    public TextGenerationPipeline? Pipeline { get; set; }

    /// <summary>The loaded GGUF state-space model (mamba/mamba2/rwkv6/rwkv7), or null for a transformer.</summary>
    public SsmLanguageModel? SsmModel { get; set; }

    /// <summary>The generation pipeline built for <see cref="SsmModel"/> on <see cref="Backend"/>.</summary>
    public SsmGenerationPipeline? SsmPipeline { get; set; }

    /// <summary>Full path of the currently-loaded GGUF file, or null if nothing is loaded.</summary>
    public string? LoadedPath { get; set; }

    /// <summary>Splice vision encoder (gemma3 / qwen2.5-vl / siglip family), or null when text-only / mllama.</summary>
    public IVlmImageEncoder? SpliceVision { get; set; }

    /// <summary>Cross-attention vision encoder (Llama-3.2-Vision), or null when text-only / splice.</summary>
    public MllamaVisionEncoder? MllamaVision { get; set; }

    /// <summary>The loaded sidecar mmproj path, or null when the model is text-only.</summary>
    public string? VisionPath { get; set; }
}
