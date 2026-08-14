namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Temporal chunking for the LTX-2.5 diffusion decoder. At 768×512×97f the stage-5 trunk carries ~2.4 M
/// tokens, so a single full-grid SwiGLU workspace is 9.3 GB and the decode cannot run untiled; every pass is instead
/// run over slices of the temporal axis, which is the axis the reference's own <c>drop_leading_frame</c> /
/// <c>pad_trailing</c> flags exist to let a caller cut.
///
/// <para>A pointwise pass (context projection, RMS-norm + modulate, SwiGLU, pixel shuffle) may be cut anywhere. A
/// neighborhood-attention pass may not: NATTEN keeps every window exactly <c>kernel</c> wide by sliding it inward at a
/// border, so rows near a naive cut attend a different set of keys than they would untiled and a seam appears at every
/// chunk boundary. <see cref="Window"/> therefore pads each chunk with a <c>kernel/2</c> halo on both sides and never
/// lets the padded window fall below the kernel — <c>IBackend.Na3d</c> collapses to <c>min(kernel, frames)</c>, so a
/// short first or last window would silently change every window inside it, not just the halo. With that padding the
/// kept rows' window starts are identical to the untiled ones, and since RoPE here is a per-token rotation (the
/// score depends only on the position difference) the halo is the only thing needed for exactness — provided the rope
/// tables are built at the window's GLOBAL frame offset, which is why the attention takes one.</para></summary>
internal static class LtxVideo25TemporalChunks
{
    /// <summary>Frames per chunk keeping a pointwise pass's workspace inside <paramref name="budgetBytes"/>.</summary>
    /// <param name="bytesPerToken">Summed width of every tensor the pass holds live at once, in bytes per token.</param>
    internal static int FramesPerChunk(long budgetBytes, long planeTokens, long bytesPerToken, int frames)
    {
        long perFrame = Math.Max(1L, planeTokens * bytesPerToken);
        return (int)Math.Clamp(budgetBytes / perFrame, 1L, frames);
    }

    /// <summary>Frames a halo-padded attention chunk owns, given the padded window <paramref name="budgetBytes"/>
    /// allows. The kernel is a floor: a window narrower than that cannot reproduce the untiled windows at all.</summary>
    internal static int AttentionFramesPerChunk(long budgetBytes, long planeTokens, long bytesPerToken, int frames,
        int kernelT)
    {
        int kernel = Math.Min(kernelT, frames);
        int window = Math.Max(kernel, FramesPerChunk(budgetBytes, planeTokens, bytesPerToken, frames));
        // A window covering the whole axis needs no halo subtracted: taking it off anyway would split a pass that
        // already fits into two chunks, which is both slower and enough to break the layer-diff harness's taps.
        return window >= frames ? frames : Math.Max(1, window - kernel / 2 * 2);
    }

    /// <summary>Halo-padded window an attention pass must see for rows <c>[start, end)</c> to come out exactly as they
    /// would in one untiled pass.</summary>
    internal static (int Start, int Frames) Window(int start, int end, int frames, int kernelT)
    {
        int kernel = Math.Min(kernelT, frames);
        int halo = kernel / 2;
        int windowStart = Math.Max(0, Math.Min(start - halo, frames - kernel));
        int windowEnd = Math.Min(frames, Math.Max(end + halo, windowStart + kernel));
        return (windowStart, windowEnd - windowStart);
    }
}
