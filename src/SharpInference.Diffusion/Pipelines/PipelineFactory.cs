using SharpInference.Core.Backends;

namespace SharpInference.Diffusion.Pipelines;

/// <summary>Factory for constructing <see cref="DiffusionPipelineBase"/> instances from on-disk model
/// artifacts. <b>Deliberately scaffolding-only as of this writing</b> — the methods below throw
/// <see cref="NotImplementedException"/> with a pointer to the design discussion that needs to happen
/// before each one can be filled in.
///
/// <para><b>Why scaffolding instead of an implementation?</b> A real factory needs to make decisions
/// that aren't yet decided in this codebase:</para>
/// <list type="number">
///   <item><b>Model-type detection.</b> Given a single-file safetensors / GGUF checkpoint on disk, how
///         do we tell whether it's SDXL vs SD3 vs Flux vs Chroma vs HiDream? This requires inspecting
///         the tensor key prefixes — every <c>*CheckpointConverter</c> has its own heuristic, but
///         there's no shared dispatch table. Either we add a <c>DetectModelType(path)</c> helper that
///         each converter contributes to, OR we make the caller pass an explicit model-type enum.</item>
///   <item><b>Component-file discovery.</b> Each pipeline takes 2-4 components (transformer, text
///         encoders, VAE). These are sometimes bundled in one safetensors file (SDXL all-in-one),
///         sometimes split across <c>text_encoder/</c>, <c>text_encoder_2/</c>, <c>transformer/</c>,
///         <c>vae/</c> folders (diffusers layout), sometimes a single transformer checkpoint plus
///         separately-downloaded encoders (Flux). The factory needs a model-aware layout resolver.</item>
///   <item><b>Tokenizer + vocab discovery.</b> Pipelines don't take tokenizers, but callers need them
///         to produce the token-id inputs. Whose responsibility is this — the factory bundles a
///         tokenizer, or the caller wires it separately?</item>
///   <item><b>Quality/dtype profile.</b> Should the factory let callers pick "fastest", "highest
///         quality", "fit-in-12GB"? <see cref="Quality.QualityProfileApplier"/> exists but the
///         per-pipeline composition isn't yet defined.</item>
///   <item><b>Caching.</b> Loading text encoders is expensive — a server would want the factory to
///         share T5/CLIP instances across pipelines. The current design hands components by value;
///         a real factory would need lifecycle management.</item>
/// </list>
///
/// <para><b>Until those are decided</b>, callers continue to construct pipelines directly via the
/// per-pipeline constructors (which take pre-loaded components). The pattern is verbose but explicit;
/// every test in <c>SharpInference.Diffusion.Tests</c> demonstrates it. The boilerplate that this
/// factory is meant to absorb lives mostly in test setup — not in production code — so the urgency
/// is lower than it appears.</para>
///
/// <para><b>Suggested next step:</b> Pick ONE pipeline family (SDXL is simplest) and write a working
/// <c>LoadSdxlFromPath(string checkpointPath, IBackend backend)</c> end-to-end. The decisions made
/// for SDXL will then constrain the shape of the rest.</para>
/// </summary>
public static class PipelineFactory
{
    /// <summary>Loads a diffusion pipeline by inspecting the checkpoint at <paramref name="path"/>
    /// and dispatching to the correct per-model loader. <b>Not implemented</b> — see class remarks
    /// for the design questions that need to be resolved before this can be built. For now, callers
    /// must construct pipelines directly via the per-pipeline constructors.</summary>
    /// <param name="path">Path to a safetensors / GGUF file or to a diffusers-layout model directory.</param>
    /// <param name="backend">Compute backend the pipeline will use for inference.</param>
    /// <exception cref="NotImplementedException">Always — this is scaffolding-only.</exception>
    public static DiffusionPipelineBase LoadAuto(string path, IBackend backend)
    {
        _ = path;
        _ = backend;
        throw new NotImplementedException(
            "PipelineFactory.LoadAuto is scaffolding only. See the class remarks for the design " +
            "decisions that need to happen before this can be implemented. Callers should construct " +
            "pipelines directly via the per-pipeline constructors (e.g., new SdxlPipeline(backend, " +
            "clipL, clipG, unet, vaeDecoder, ...)) — every test in SharpInference.Diffusion.Tests " +
            "demonstrates the pattern.");
    }
}
