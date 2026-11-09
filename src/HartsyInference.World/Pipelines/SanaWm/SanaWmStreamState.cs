using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>Which dtype a SANA-WM stream holds its softmax windows and key tails in between chunks. BF16 is upstream's own
/// storage and halves those states (about 0.9 GB at 720p); each pass then widens them again, about 40 ms per pass on a
/// 4090 (+9% chunk time, measured). F32 holds them as computed and is faster. So the narrow form is for a card with no
/// room for the wide one, which <c>auto</c> takes to be a device of 16 GiB or less, and F32 is for the rest.</summary>
public static class SanaWmStreamState
{
    /// <summary>The largest device, in bytes, on which <c>auto</c> holds the state in BF16: a 12 GB card runs out of memory
    /// decoding chunk 1 with F32 state, a 24 GB card has room and keeps the faster form.</summary>
    public const long AutoBf16AtOrBelowBytes = 16L << 30;

    /// <summary>The storage dtype for <paramref name="mode"/>: <c>auto</c> (or null), <c>f32</c> or <c>bf16</c>. Null in the
    /// result means the state is held as computed (F32). A device whose memory is unknown (the CPU backend) is not small.</summary>
    /// <exception cref="ArgumentException">An unknown mode.</exception>
    public static DType? Resolve(IBackend backend, string? mode)
    {
        ArgumentNullException.ThrowIfNull(backend);
        switch ((mode ?? "auto").Trim().ToLowerInvariant())
        {
            case "f32":
                return null;
            case "bf16":
                return DType.BF16;
            case "" or "auto":
                long total = backend.GetVramInfo().TotalBytes;
                return total > 0 && total <= AutoBf16AtOrBelowBytes ? DType.BF16 : null;
            default:
                throw new ArgumentException($"Unknown SANA-WM stream state mode '{mode}'; use auto, f32 or bf16.", nameof(mode));
        }
    }
}
