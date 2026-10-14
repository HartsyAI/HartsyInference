namespace HartsyInference.Vulkan;

/// <summary>When causal prefill attention may take the cooperative-matrix-2 kernel.</summary>
/// <remarks>The causal and sliding-window mask is computed inside <c>sdpa_flash_cm2_causal</c> from the query position and
/// the key column, so no mask is built on the host or uploaded: the cost that used to scale with <c>sq * kvLen</c> (4 bytes
/// per pair, uploaded per layer) is gone. The bound below is kept as a shape policy, not a memory guard. The text path
/// prefills the whole prompt in one call (<c>MaxPrefillChunk</c> is not applied there), so <c>sq</c> is the full prompt
/// length. Above <see cref="MaxMaskBytes"/> (8K by 8K pairs) the cm2 path is not taken and the scalar kernel runs, as it
/// does on main. Lifting it would change which kernel longer prompts run (F32 scalar to F16 cm2), so it needs its own
/// numerics check at 16K and above first.</remarks>
internal static class FlashCm2PrefillPolicy
{
    /// <summary>The validated shape bound, in the units of the former host mask: 256 MiB, which is 8K query rows by 8K keys.</summary>
    internal const long MaxMaskBytes = 256L << 20;

    /// <summary>Query-by-key area in the units of the former F32 host mask: 4 bytes per (query, key) pair.</summary>
    internal static long MaskBytes(long sq, long kvLen) => 4L * sq * kvLen;

    /// <summary>Whether the cm2 causal path serves this shape: positive dimensions and at most <see cref="MaxMaskBytes"/>.</summary>
    internal static bool MaskFits(long sq, long kvLen) => sq > 0 && kvLen > 0 && MaskBytes(sq, kvLen) <= MaxMaskBytes;
}
