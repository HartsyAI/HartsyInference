using HartsyInference.Core.Backends;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Multimodal;
using HartsyInference.LLM.Ssm;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Engine.Services;

/// <summary>One compute device's loaded LLM state owned by <see cref="TextService"/>. A single model/backend/pipeline is not safe for concurrent use, so each slot carries its own lock — but two different slots (e.g. cuda:0 + cpu) can generate at the same time. Exactly one of <see cref="Model"/> / <see cref="SsmModel"/> is set once loaded.</summary>
internal sealed class TextDeviceSlot
{
    /// <summary>Serializes generation on this slot; different slots run concurrently.</summary>
    public SemaphoreSlim Lock { get; } = new SemaphoreSlim(1, 1);

    /// <summary>The compute backend (CPU/CUDA) bound to this slot's device. For a layer-split load this is the LAST stage's backend (logits/sampling live there).</summary>
    public IBackend? Backend { get; set; }

    /// <summary>The other stage backends of a layer-split load (everything except <see cref="Backend"/>). Slot-owned: disposed with the slot, exactly like <see cref="Backend"/>.</summary>
    public List<IBackend>? ExtraStageBackends { get; set; }

    /// <summary>The active layer-split plan, or null for a single-device load.</summary>
    public LlmPlacement? Placement { get; set; }

    /// <summary>The loaded GGUF transformer model, or null when nothing (or an SSM model) is loaded.</summary>
    public GgufLanguageModel? Model { get; set; }

    /// <summary>The generation pipeline built for <see cref="Model"/> on <see cref="Backend"/>.</summary>
    public TextGenerationPipeline? Pipeline { get; set; }

    /// <summary>Tensor-parallel transformer (<c>TensorParallelDegree</c> &gt; 1), or null. Rank backends reuse the layer-split fields: <see cref="Backend"/> = rank 0 (logits/sampling), <see cref="ExtraStageBackends"/> = ranks 1.. — so every existing slot-backend disposal path covers TP unchanged.</summary>
    public TensorParallelTransformer? TpTransformer { get; set; }

    /// <summary>The TP collective communicator (owned; disposed with the slot's model).</summary>
    public ICollectiveComm? TpComm { get; set; }

    /// <summary>The TP checkpoint (weight dict + mmap + tokenizer/template) — must outlive <see cref="TpTransformer"/>, whose rank slices were copied from (and whose embed gathers read) it.</summary>
    public GgufLanguageModel.TpCheckpoint? TpCheckpoint { get; set; }

    /// <summary>The loaded GGUF state-space model (mamba/mamba2/rwkv6/rwkv7), or null for a transformer.</summary>
    public SsmLanguageModel? SsmModel { get; set; }

    /// <summary>The generation pipeline built for <see cref="SsmModel"/> on <see cref="Backend"/>.</summary>
    public SsmGenerationPipeline? SsmPipeline { get; set; }

    /// <summary>The loaded DeepSeek-V4.1 host reference model, or null. Its <see cref="Pipeline"/> is built over the model's own generation adapter; the slot's <see cref="Backend"/> is a CPU backend.</summary>
    public DeepSeekV41TextModel? DeepSeekV41 { get; set; }

    /// <summary>Full path of the currently-loaded model (a GGUF file or a Hugging Face directory), or null if nothing is loaded.</summary>
    public string? LoadedPath { get; set; }

    /// <summary>Splice vision encoder (gemma3 / qwen2.5-vl / siglip family), or null when text-only / mllama.</summary>
    public IVlmImageEncoder? SpliceVision { get; set; }

    /// <summary>Cross-attention vision encoder (Llama-3.2-Vision), or null when text-only / splice.</summary>
    public MllamaVisionEncoder? MllamaVision { get; set; }

    /// <summary>The loaded sidecar mmproj path, or null when the model is text-only.</summary>
    public string? VisionPath { get; set; }

    /// <summary>Per-key retained KV sequences for opt-in prefix-cache reuse against <see cref="Pipeline"/>; null
    /// until the first such request on this slot. Bound to <see cref="Model"/>'s weights — disposed and cleared
    /// whenever the slot's model/backend is torn down (reload, explicit unload), never carried over to a
    /// different model.</summary>
    public RetainedSequenceStore? PrefixCache { get; set; }

    /// <summary>The effective <see cref="IBackend.CacheWeightCasts"/> value in force since this slot's backend was
    /// created (whichever request's <c>TextRequest.CacheWeightCasts</c>, if any, happened to be the first on this
    /// slot — or the backend's own default if none did). Read-only after creation: a later request on an
    /// already-loaded slot cannot change it without a reload, so this is what <see cref="TextService.LoadInto"/>'s
    /// early-return compares a mismatched later request against, to log that its own value was ignored.</summary>
    public bool? CacheWeightCastsApplied { get; set; }

    /// <summary>The effective <c>includeRedundantSplits</c> value <see cref="TextService.LoadInto"/> preloaded
    /// weights with for this slot (the single-device path's own <c>TextRequest.PreloadRedundantWeightSplits</c>
    /// decision, or <c>false</c> unconditionally for the sharded/tensor-parallel paths, which never read that
    /// field). Same read-only-after-creation caveat as <see cref="CacheWeightCastsApplied"/>.</summary>
    public bool? PreloadRedundantWeightSplitsApplied { get; set; }

    /// <summary>Which load-time settings <see cref="TextService.LoadInto"/> has already logged a mismatch for on
    /// this slot (by setting name) — a request's own value keeps being ignored every subsequent call on an
    /// already-loaded slot, so without this a long voice call would repeat the same debug line every turn. Reset
    /// by <c>UnloadSlot</c>, so a reload gets a fresh warning if it mismatches again.</summary>
    public HashSet<string> LoggedSettingMismatches { get; } = [];
}
