using HartsyInference.Core.Tensors;
using HartsyInference.Vision.CrossFrame;

namespace HartsyInference.Vision.Dinov3;

/// <summary>
/// The DINOv3 RoPE tables of the last shape an encoder ran with it, for a caller that runs one shape again and again (a video stream): each table is then
/// built on the host, and uploaded to a device, once instead of once per frame. Another batch, grid, frame count or preset replaces them. Use one cache per
/// stream: it is not thread-safe, and its tables stay with the backend that stream runs on.
/// </summary>
public sealed class Dinov3RopeCache : IDisposable
{
    private Key _key;
    private Tensor? _cos, _sin, _crossCos, _crossSin;
    private bool _disposed;

    /// <summary>How many times the tables were built: once for a stream of one shape.</summary>
    public int Builds { get; private set; }

    // The tables of a block, [Batch, Prefix + patches, HeadDim], and with cross-frame blocks theirs: the joint sequence's of CrossFrames frames (the prefix
    // rows, then the patch rows once per frame), or with zero frames the patch rows alone, which a cached frame's patches take. The cache owns all four.
    internal (Tensor Cos, Tensor Sin, Tensor? CrossCos, Tensor? CrossSin) Get(in Key key)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_cos is not null && _key == key) return (_cos, _sin!, _crossCos, _crossSin);
        Release();
        (_cos, _sin) = Dinov3RopeTables.Build(key.Batch, key.Prefix, key.GridH, key.GridW, key.HeadDim, key.RoundPeriodsToBf16, key.NormalizeByLongSide);
        if (key.CrossFrames is int frames)
        {
            (Tensor cos, Tensor sin) = Dinov3RopeTables.Build(1, 0, key.GridH, key.GridW, key.HeadDim, key.RoundPeriodsToBf16, key.NormalizeByLongSide);
            if (frames == 0)
            {
                (_crossCos, _crossSin) = (cos, sin);
            }
            else
            {
                using (cos)
                using (sin)
                {
                    _crossCos = CrossFrameTokens.TileRope(cos, frames, key.Prefix, 1f);
                    _crossSin = CrossFrameTokens.TileRope(sin, frames, key.Prefix, 0f);
                }
            }
        }

        _key = key;
        Builds++;
        return (_cos, _sin!, _crossCos, _crossSin);
    }

    public void Dispose()
    {
        _disposed = true;
        Release();
    }

    private void Release()
    {
        _cos?.Dispose();
        _sin?.Dispose();
        _crossCos?.Dispose();
        _crossSin?.Dispose();
        _cos = _sin = _crossCos = _crossSin = null;
    }

    // CrossFrames is null when the run has no cross-frame block.
    internal readonly record struct Key(int Batch, int Prefix, int GridH, int GridW, int HeadDim, bool RoundPeriodsToBf16, bool NormalizeByLongSide,
        int? CrossFrames);
}
