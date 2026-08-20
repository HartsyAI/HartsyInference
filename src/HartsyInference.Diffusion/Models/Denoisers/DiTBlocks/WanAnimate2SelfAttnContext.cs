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

    /// <summary>The current block's cached driving K, <c>[refSeq, dim]</c>, stored <b>pre-RoPE</b> — the driving
    /// RoPE table is applied on read, so a resolution change can never re-use a stale offset. Reassigned by the
    /// transformer before each block.</summary>
    public Tensor? DrivingK { get; set; }

    /// <summary>The current block's cached driving V, <c>[refSeq, dim]</c> (values are never rotated).</summary>
    public Tensor? DrivingV { get; set; }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        RefCos.Dispose();
        RefSin.Dispose();
        KeyBuffer.Dispose();
        ValueBuffer.Dispose();
        LogScaleBiasGen?.Dispose();
        LogScaleBiasSpliced?.Dispose();
    }
}

/// <summary>Receives a block's self-attention K/V <b>before</b> RoPE, for the Animate-2 driving prepass. Caller-owned
/// (so CFG branch threads can't collide) and caller-disposed.</summary>
public sealed class WanAnimate2KvCapture
{
    /// <summary>Post-QK-norm, pre-RoPE self-attention keys <c>[S, dim]</c>.</summary>
    public Tensor? K { get; internal set; }

    /// <summary>Self-attention values <c>[S, dim]</c>.</summary>
    public Tensor? V { get; internal set; }
}
