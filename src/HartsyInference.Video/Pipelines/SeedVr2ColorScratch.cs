using System.Runtime.InteropServices;

namespace HartsyInference.Video.Pipelines;

/// <summary>Native scratch for the SeedVR2 colour passes, sized once per restore so the frame loop allocates nothing.</summary>
/// <remarks>Every pass works in planes of one frame, so the buffers are sized by pixel count and reused for each frame. Owned
/// native memory (not managed arrays) follows the inference rule: no managed allocation on the hot path.</remarks>
public sealed unsafe class SeedVr2ColorScratch : IDisposable
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
    private float* _floats;
    private int* _order;
    private bool _disposed;

    /// <summary>Allocates scratch for frames of <paramref name="pixels"/> pixels each.</summary>
    public SeedVr2ColorScratch(int pixels)
    {
        if (pixels <= 0)
            throw new ArgumentOutOfRangeException(nameof(pixels));
        _pixels = pixels;
        _floats = (float*)NativeMemory.AlignedAlloc((nuint)((long)PlanesTotal * pixels * sizeof(float)), 64);
        _order = (int*)NativeMemory.AlignedAlloc((nuint)((long)pixels * sizeof(int)), 64);
    }

    /// <summary>Pixels per frame this scratch was sized for.</summary>
    public int Pixels => _pixels;

    /// <summary>The region's planes: <c>3 × Pixels</c> floats for an RGB or LAB region, <c>Pixels</c> for a single plane.</summary>
    public Span<float> Get(Region region)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int offset = RegionOffsetPlanes[(int)region] * _pixels;
        return new Span<float>(_floats + offset, PlanesPerRegion[(int)region] * _pixels);
    }

    /// <summary>The sort order of one histogram match, one index per pixel.</summary>
    public Span<int> Order
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return new Span<int>(_order, _pixels);
        }
    }

    /// <summary>Frees the native buffers.</summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        NativeMemory.AlignedFree(_floats);
        NativeMemory.AlignedFree(_order);
        _floats = null;
        _order = null;
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
