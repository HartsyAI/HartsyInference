namespace HartsyInference.Vulkan;

/// <summary>When causal prefill attention may take the cooperative-matrix-2 kernel, and what that costs on the host.</summary>
/// <remarks>The kernel takes its causal and sliding-window mask as a host-built F32 tile of <c>sq x kvLen</c> values.
/// The tile is built on the host once per prefill and uploaded again for every layer, so it costs 4 bytes per
/// (query, key) pair per layer: 64 MiB at 4K, 256 MiB at 8K, 4 GiB at 32K. The text path prefills the whole prompt in
/// one call (<c>MaxPrefillChunk</c> is not applied there), so <c>sq</c> is the full prompt length. Above
/// <see cref="MaxMaskBytes"/> the cm2 path is not taken and the scalar kernel runs, as it does on main.
/// The proper fix is a device-side mask with no host upload. That is a follow-up.</remarks>
internal static class FlashCm2PrefillPolicy
{
    /// <summary>The largest host mask the cm2 path builds: 256 MiB, which is 8K query rows by 8K keys.</summary>
    internal const long MaxMaskBytes = 256L << 20;

    /// <summary>Bytes of the F32 causal mask for <paramref name="sq"/> query rows over <paramref name="kvLen"/> keys.</summary>
    internal static long MaskBytes(long sq, long kvLen) => 4L * sq * kvLen;

    /// <summary>Whether the cm2 path may build the mask for this shape: positive dimensions and at most <see cref="MaxMaskBytes"/>.</summary>
    internal static bool MaskFits(long sq, long kvLen) => sq > 0 && kvLen > 0 && MaskBytes(sq, kvLen) <= MaxMaskBytes;
}
