namespace HartsyInference.Core.Backends;

/// <summary>Describes what operations a backend supports. Used by the runtime to determine whether to route an operation to this backend or fall back.</summary>
public sealed class BackendCapabilities
{
    /// <summary>Whether this backend supports FP32 operations.</summary>
    public bool SupportsF32 { get; init; } = true;

    /// <summary>Whether this backend supports FP16 operations.</summary>
    public bool SupportsF16 { get; init; }

    /// <summary>Whether this backend supports BF16 operations.</summary>
    public bool SupportsBF16 { get; init; }

    /// <summary>Whether this backend supports quantized (Q8_0, Q4_K) operations.</summary>
    public bool SupportsQuantized { get; init; }

    /// <summary>Whether this backend supports Conv2D.</summary>
    public bool SupportsConv2D { get; init; }

    /// <summary>Whether Conv2D internally bands its im2col workspace (caps the largest single allocation at <see cref="Im2ColWorkspaceCapBytes"/>); lets VRAM planners (e.g. full-res VAE decode) assume a bounded per-conv workspace instead of the naive inCh·k²·outH·outW blow-up.</summary>
    public bool BandsIm2Col { get; init; }

    /// <summary>Largest single im2col workspace Conv2D will allocate when <see cref="BandsIm2Col"/> is true.</summary>
    public long Im2ColWorkspaceCapBytes { get; init; }

    /// <summary>Whether this backend supports scaled dot-product attention.</summary>
    public bool SupportsSdpa { get; init; }

    /// <summary>Whether this backend supports FFT / STFT for audio processing.</summary>
    public bool SupportsFft { get; init; }

    /// <summary>Maximum tensor dimensions supported.</summary>
    public int MaxRank { get; init; } = 6;

    /// <summary>Descriptive name of this backend.</summary>
    public required string Name { get; init; }
}
