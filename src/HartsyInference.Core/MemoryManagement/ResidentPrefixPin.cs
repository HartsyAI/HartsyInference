namespace HartsyInference.Core.MemoryManagement;

/// <summary>Cross-generation state for a denoiser's resident block prefix: how many blocks the sizing pinned, the
/// token load it reserved headroom for, and whether those weights are still on the device.</summary>
/// <remarks>A pipeline holds one of these per denoiser and hands it to every <see cref="BlockStreamingScope"/> it
/// opens. Without it the prefix is re-sized every generation and drifts with pool slack (measured on LTX-2:
/// 12 → 10 → 9 blocks across three runs), which moves the streaming boundary and re-uploads gigabytes. The
/// <see cref="Resident"/>/<see cref="TailEvicted"/> flags are settable because the phases that displace the prefix —
/// a text-encoder miss, a VAE decode, a two-stage upsampler — live outside the denoise scope.</remarks>
public sealed class ResidentPrefixPin
{
    /// <summary>Blocks the sizing pinned, or <c>-1</c> before the first sizing.</summary>
    public int PinnedBlocks { get; internal set; } = -1;

    /// <summary>Token load the pinned count reserved headroom for, or <c>-1</c> before the first sizing.</summary>
    public long SizedTokens { get; internal set; } = -1;

    /// <summary>True while the pinned prefix's weights are still on the device from a previous generation.</summary>
    public bool Resident { get; set; }

    /// <summary>Set when a later phase freed only the prefix's tail: the pin stands, but the pool must be trimmed
    /// before the next generation tops it back up.</summary>
    public bool TailEvicted { get; set; }
}
