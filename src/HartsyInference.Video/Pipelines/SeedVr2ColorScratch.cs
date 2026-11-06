using HartsyInference.Core.Memory.Host;

namespace HartsyInference.Video.Pipelines;

/// <summary>Native scratch for the SeedVR2 colour passes, sized once per restore so the frame loop allocates nothing.</summary>
/// <remarks>Every pass works in planes of one frame, so the buffers are sized by pixel count and reused for each frame. Owned
/// native memory (not managed arrays) follows the inference rule: no managed allocation on the hot path.</remarks>
public sealed class SeedVr2ColorScratch : IDisposable
{
    /// <summary>Named regions of the scratch, in float planes of one frame each (a 3 means three planes).</summary>
    public enum Region
    {
        /// <summary>The model's frame after the wavelet recombination, RGB.</summary>
        Content = 0,
        /// <summary>One blur pass's ping-pong buffer.</summary>
        Blur,
        /// <summary>The model's low-frequency plane.</summary>
        ModelLow,
        /// <summary>The input's low-frequency plane.</summary>
        InputLow,
        /// <summary>The content in CIELAB, L/a/b.</summary>
        ContentLab,
        /// <summary>The reference in CIELAB, L/a/b.</summary>
        ReferenceLab,
        /// <summary>The histogram-matched CIELAB, L/a/b; also the LAB→RGB source.</summary>
        Matched,
        /// <summary>Sort keys for one histogram match.</summary>
        SortKeys,
        /// <summary>The reference values, sorted for one histogram match.</summary>
        SortRef,
        /// <summary>One frame's RGB copied out of the clip-major output.</summary>
        FrameModel,
        /// <summary>One frame's RGB reference, copied out of the resized input.</summary>
        FrameInput,
    }

    private static readonly int[] PlanesPerRegion = [3, 1, 1, 1, 3, 3, 3, 1, 1, 3, 3];
    private static readonly int[] RegionOffsetPlanes = ComputeOffsets();
    private static readonly int PlanesTotal = PlanesPerRegion.Sum();

    private readonly int _pixels;
    private readonly NativeBuffer _floats;
    private readonly NativeBuffer _order;

    /// <summary>Allocates scratch for frames of <paramref name="pixels"/> pixels each.</summary>
    /// <remarks>Both buffers are finalizer-backed <see cref="NativeBuffer"/>s, so a scratch that is never disposed, or whose
    /// second allocation fails, still releases its memory.</remarks>
    public SeedVr2ColorScratch(int pixels)
    {
        if (pixels <= 0)
            throw new ArgumentOutOfRangeException(nameof(pixels));
        _pixels = pixels;
        _floats = new NativeBuffer((nuint)((long)PlanesTotal * pixels * sizeof(float)));
        _order = new NativeBuffer((nuint)((long)pixels * sizeof(int)));
    }

    /// <summary>Pixels per frame this scratch was sized for.</summary>
    public int Pixels => _pixels;

    /// <summary>The region's planes: <c>3 × Pixels</c> floats for an RGB or LAB region, <c>Pixels</c> for a single plane.</summary>
    public Span<float> Get(Region region)
    {
        int offset = RegionOffsetPlanes[(int)region] * _pixels;
        return _floats.AsSpan<float>().Slice(offset, PlanesPerRegion[(int)region] * _pixels);
    }

    /// <summary>The sort order of one histogram match, one index per pixel.</summary>
    public Span<int> Order => _order.AsSpan<int>().Slice(0, _pixels);

    /// <summary>Frees the native buffers. Idempotent.</summary>
    public void Dispose()
    {
        _floats.Dispose();
        _order.Dispose();
    }

    private static int[] ComputeOffsets()
    {
        int[] offsets = new int[PlanesPerRegion.Length];
        int running = 0;
        for (int i = 0; i < PlanesPerRegion.Length; i++)
        {
            offsets[i] = running;
            running += PlanesPerRegion[i];
        }
        return offsets;
    }
}
