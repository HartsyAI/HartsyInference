using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Per-forward driving-stream state for Wan-Animate-2's frame-local ("Sparse-Ref") self-attention, handed to
/// <see cref="WanVideoBlock.Forward"/>. Generation latent frame <c>j</c> attends every generation token plus ONLY
/// driving frame <c>j-1</c>; frame 0 is the reference-image slot and sees no driving tokens at all.
///
/// <para>The key/value splice buffers are allocated once and reused by all 40 blocks and every frame: the generation
/// half is written once per block and only the <c>hw</c>-token driving tail is rewritten per frame.</para></summary>
public sealed class WanAnimate2SelfAttnContext : IDisposable
{
    private int _disposed;

    /// <summary>RoPE table for the driving stream, built with the per-axis offsets <c>(t=1, h=0, w=genGridW)</c>.</summary>
    public required Tensor RefCos { get; init; }

    /// <inheritdoc cref="RefCos"/>
    public required Tensor RefSin { get; init; }

    /// <summary>Generation latent frames (the driving stream always has exactly one fewer).</summary>
    public required int GenFrames { get; init; }

    /// <summary>Tokens per latent frame (<c>gridH · gridW</c>) — the attention band stride for both streams.</summary>
    public required int TokensPerFrame { get; init; }

    /// <summary>Additive score bias over the <c>[hw, 2hw)</c> key band (generation latent frame 1) for the
    /// frame-0 query block, <c>[hw, S]</c>. Null when <c>log_scale</c> is 0 — the base build, which then takes the
    /// unmasked attention path.</summary>
    public Tensor? LogScaleBiasGen { get; init; }

    /// <summary>The same bias for the spliced key sequence of frames <c>&gt; 0</c>, <c>[hw, S + hw]</c>.</summary>
    public Tensor? LogScaleBiasSpliced { get; init; }

    /// <summary>Splice buffer <c>[1, heads, S + hw, headDim]</c>: generation keys followed by one driving frame.</summary>
    public required Tensor KeyBuffer { get; init; }

    /// <inheritdoc cref="KeyBuffer"/>
    public required Tensor ValueBuffer { get; init; }

    /// <summary>The current block's cached driving <b>self-attention input</b> <c>[refSeq, dim]</c> — the normed,
    /// modulated hidden state the <c>k</c>/<c>v</c> projections consume. K and V are re-projected from it on read
    /// (half the memory of caching both, for one extra projection pair per block), and the driving RoPE table is
    /// still applied after that projection, so the cache remains pre-RoPE. F32, or BF16 when the caller opted into
    /// the halved cache. Reassigned by the transformer before each block.</summary>
    public Tensor? DrivingInput { get; set; }

    /// <summary>Scales the driving stream's V before it is spliced. 1.0 is the trained behaviour. Mirrors
    /// ComfyUI's <c>pose_strength</c>; like it, 0.0 does NOT mute the branch, because K still occupies the
    /// softmax denominator.
    /// <para>⚠️ This was introduced to compensate for long clips not following the driver. There was no such
    /// defect: the clips it was reaching for were the base checkpoint sampled at the distillation build's 6 steps
    /// and cfg 1, which renders hazy at any length and only looks acceptable on a short one. At 40 steps and
    /// guidance 3.0 the driver reaches the output at the full 77-frame length with this left at 1.0. Keep it at
    /// 1.0 unless a user deliberately wants over- or under-driven motion.</para></summary>
    public float PoseStrength { get; set; } = 1.0f;

    /// <summary>Releases the splice buffers only. The rope tables, the log-scale bias and the driving input are all
    /// owned by longer-lived caches and are borrowed here for the duration of one forward.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        KeyBuffer.Dispose();
        ValueBuffer.Dispose();
    }
}

/// <summary>Receives a block's self-attention input <b>before</b> the k/v projections, for the Animate-2 driving
/// prepass. Caller-owned (so CFG branch threads can't collide) and caller-disposed.</summary>
public sealed class WanAnimate2KvCapture
{
    /// <summary>The normed, modulated self-attention input <c>[S, dim]</c> — what the driving cache actually stores.</summary>
    public Tensor? Input { get; internal set; }

    /// <summary>Also hand back the projected K/V, which the cache no longer stores. Off by default: this exists so a
    /// test can compare the re-projected pair against the pair computed inline by the prepass.</summary>
    public bool CaptureProjected { get; init; }

    /// <summary>Post-QK-norm, pre-RoPE self-attention keys <c>[S, dim]</c>; null unless <see cref="CaptureProjected"/>.</summary>
    public Tensor? K { get; internal set; }

    /// <summary>Self-attention values <c>[S, dim]</c>; null unless <see cref="CaptureProjected"/>.</summary>
    public Tensor? V { get; internal set; }
}
