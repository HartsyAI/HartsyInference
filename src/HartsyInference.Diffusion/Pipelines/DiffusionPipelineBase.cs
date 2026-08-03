using HartsyInference.Core.Backends;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Shared scaffolding for every diffusion pipeline: holds the compute <see cref="IBackend"/>, implements the idempotent thread-safe disposal flag, and exposes <see cref="ThrowIfDisposed"/> so subclasses don't each reinvent the pattern.
/// <para><b>Component ownership.</b> By convention pipelines do NOT own the text encoders, transformers/UNets, or VAE halves passed into them on construction — those are <i>shared</i> resources (e.g., one CLIP-L shared by SDXL base + refiner) and stay alive across pipeline disposes. Subclasses that allocate pipeline-internal state (e.g., a cached scheduler instance, lazy tensors) should override <see cref="DisposeCore"/> to release it.</para>
/// <para><b>Why there's no <c>DenoiseLoopRunner</c> on this base.</b> The audit that produced this class
/// flagged the per-pipeline denoise loop scaffolding (~100 lines × 17 pipelines) as a duplication
/// candidate. After migrating every pipeline, the conclusion was the opposite: the loops are <i>not</i>
/// duplicating — they encode the model-specific quirks that distinguish each architecture, and an
/// abstraction would either need so many parameters that it's worse than inline code, or hide the
/// quirks where they're hardest to debug. Examples of per-model loop variation that resist a shared
/// runner:</para>
/// <list type="bullet">
///   <item><b>Flux</b>: streaming-block controller threaded through every step; FLUX.1 Tools control
///         concat into the transformer input; per-step preview-unpack of packed latent.</item>
///   <item><b>Z-Image</b>: non-standard CFG formula (<c>cond + cfg*(cond-uncond)</c> instead of
///         <c>uncond + cfg*(cond-uncond)</c>) plus mandatory velocity negation per step.</item>
///   <item><b>Lumina 2.0</b>: timestep inversion (<c>1 - sigma</c>) before the transformer call.</item>
///   <item><b>F-Lite</b>: <i>no scheduler.Step at all</i> — custom dynamic-shift Euler integrator
///         operating on a manual accumulator buffer.</item>
///   <item><b>Anima</b>: Cosmos-convention <c>t/1000</c> timestep normalization unique to that family.</item>
///   <item><b>SDXL</b>: refiner step-swap mid-loop, ControlNet residual fan-in, IP-Adapter gating with
///         start/end fraction windows, masked-inpaint per-step source re-noising.</item>
/// </list>
/// <para>What <i>did</i> deduplicate cleanly: <see cref="Utilities.CfgHelper"/> (slice + apply CFG),
/// <see cref="Utilities.DtypeCastHelper"/> (cast at activation boundaries),
/// <see cref="Utilities.Img2ImgSetup"/> (validate source + compute startStep),
/// <see cref="Schedulers.SchedulerFactory"/> (name → scheduler). Those collapse boilerplate without
/// hiding model-specific behavior. The denoise loop itself stays inline.</para>
/// </summary>
public abstract class DiffusionPipelineBase : IDisposable
{
    /// <summary>Compute backend used by every model component the pipeline routes through unless a component
    /// backend below overrides it. Injected at construction, immutable for the pipeline's lifetime.</summary>
    protected IBackend Backend { get; }

    /// <summary>Backend the prompt/text encoders run on; defaults to <see cref="Backend"/>. Settable at
    /// construction only (init) — cached pipelines are keyed by placement, so a live pipeline never re-places.
    /// Safe to point at another GPU because encoder→denoiser handoffs host-materialize the conditioning (the
    /// pre-loop <c>DataPointer</c> sweeps are the load-bearing boundary).</summary>
    public IBackend TextEncoderBackend { get; init; }

    /// <summary>Backend the VAE encode/decode runs on; defaults to <see cref="Backend"/>. Same host-side
    /// boundary argument as <see cref="TextEncoderBackend"/> (latents cross via host tensors).</summary>
    public IBackend VaeBackend { get; init; }

    /// <summary>Second backend to run the CFG uncond branch on, concurrent with cond on <see cref="Backend"/>;
    /// null (unlike <see cref="TextEncoderBackend"/>/<see cref="VaeBackend"/>, which default to
    /// <see cref="Backend"/>) means CFG-branch parallelism is off — the denoise loop runs cond/uncond
    /// sequentially on <see cref="Backend"/> as before. Settable at construction only (init).</summary>
    public IBackend? CfgParallelBackend { get; init; }

    /// <summary>Second backend to run the DiT's tail block range on for VRAM-pooling sharding (Phase 8); null
    /// (unlike <see cref="TextEncoderBackend"/>/<see cref="VaeBackend"/>) means it's off — the denoise loop runs
    /// the whole DiT on <see cref="Backend"/> as before. Unlike <see cref="CfgParallelBackend"/> this SPLITS the
    /// block range instead of replicating weights, so the win is pooled VRAM, not latency. Settable at
    /// construction only (init); pairs with <see cref="DitShardSplitBlock"/>.</summary>
    public IBackend? DitShardBackend { get; init; }

    /// <summary>Block index at which the DiT's block loop splits when <see cref="DitShardBackend"/> is set:
    /// <see cref="Backend"/> runs <c>[0, DitShardSplitBlock)</c>, <see cref="DitShardBackend"/> runs
    /// <c>[DitShardSplitBlock, BlockCount)</c>. Meaningless when <see cref="DitShardBackend"/> is null.</summary>
    public int DitShardSplitBlock { get; init; }

    private int _disposed;

    /// <summary>Initializes the base with the compute backend. Subclasses pass <c>backend</c> through from their constructor and assign their own component fields after this call.</summary>
    protected DiffusionPipelineBase(IBackend backend)
    {
        Backend = backend ?? throw new ArgumentNullException(nameof(backend));
        TextEncoderBackend = backend;
        VaeBackend = backend;
    }

    /// <summary>Throws <see cref="ObjectDisposedException"/> if the pipeline has already been disposed. Every public <c>GenerateXxx</c> entry point should call this on first line.</summary>
    protected void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    /// <summary>Override in subclasses that hold pipeline-internal disposable state (caches, lazy buffers). The base call is idempotent and thread-safe; override is invoked exactly once even under concurrent <see cref="Dispose"/> calls. Do NOT dispose injected components (text encoders, UNet, VAE) here — those are shared resources owned by the caller.</summary>
    protected virtual void DisposeCore() { }

    /// <summary>Marks the pipeline disposed and runs the subclass <see cref="DisposeCore"/> hook exactly once. Subsequent calls are no-ops. Does NOT dispose the backend or injected model components — by design (see class remarks).</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        DisposeCore();
        GC.SuppressFinalize(this);
    }
}
