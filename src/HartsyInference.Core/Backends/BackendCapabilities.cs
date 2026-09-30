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

    /// <summary>Whether a block-scaled weight (NVFP4/MXFP4/MXFP8 with its scale tensor on <c>QuantInfo</c>) multiplies natively here, so a loader should keep it packed rather than fold it to fp8 or F16.</summary>
    public bool NativeBlockScaledGemm { get; init; }

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

    /// <summary>Who made the device, for the decisions whose answer is per-vendor rather than per-API — cooperative
    /// matrix reliability, subgroup width, kernel tuning constants. <see cref="GpuVendor.Software"/> marks a CPU
    /// implementation of a GPU API, which is useful for correctness and must never be reported as hardware.</summary>
    public GpuVendor Vendor { get; init; }

    /// <summary>The device's own name, as the driver reports it ("NVIDIA GeForce RTX 4090"). <see cref="Name"/> is a
    /// description of the BACKEND and is free to say whatever reads well; this is the device, for evidence rows.</summary>
    public string DeviceName { get; init; } = "";

    /// <summary>Total device memory in bytes; 0 when not a device backend.</summary>
    /// <remarks>Distinct from a free/total query, which is a runtime measurement: this is the card's size, known at
    /// construction, and is what a VRAM tier is resolved from.</remarks>
    public long TotalVramBytes { get; init; }
}
