using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Streaming;

/// <summary>The last few frames a causal temporal operator has seen, kept as its left context for the next chunk: the
/// input tail of a temporal convolution in a chunk-causal transformer, or of a causal convolution in a video VAE. Holds at
/// most <see cref="TailFrames"/> frames. A chunk shorter than that keeps the newest frames of the previous tail in front of
/// it, so the tail is always the newest frames of the whole stream.</summary>
/// <remarks>Frames run along <c>axis</c>, each taking <c>positionsPerFrame</c> positions: 1 for a <c>[B, C, T, H, W]</c>
/// activation on axis 2, the tokens of a frame for frame-major token rows on axis 0. The cache copies what it keeps, so the
/// chunk stays the caller's.</remarks>
public sealed class FrameTailCache : IStreamingState
{
    private readonly StreamingTensorSlot _slot;
    private readonly IBackend _backend;
    private readonly int _axis;
    private readonly int _positionsPerFrame;
    private int _heldFrames;
    private int _stagedFrames;

    /// <summary>Creates an empty tail of up to <paramref name="tailFrames"/> frames.</summary>
    public FrameTailCache(IBackend backend, int tailFrames, int axis, int positionsPerFrame = 1)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentOutOfRangeException.ThrowIfLessThan(tailFrames, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(axis);
        ArgumentOutOfRangeException.ThrowIfLessThan(positionsPerFrame, 1);
        _backend = backend;
        _slot = new StreamingTensorSlot(backend);
        TailFrames = tailFrames;
        _axis = axis;
        _positionsPerFrame = positionsPerFrame;
    }

    /// <summary>The most frames the tail holds.</summary>
    public int TailFrames { get; }

    /// <summary>Frames in the committed tail: 0 before the first commit, and below <see cref="TailFrames"/> while the stream is
    /// shorter than it.</summary>
    public int HeldFrames => _heldFrames;

    /// <summary>The committed tail, oldest frame first, or null before the first commit. Borrowed: valid until the next commit
    /// or reset.</summary>
    public Tensor? Current => _slot.Current;

    /// <summary>Stages the tail after <paramref name="chunk"/>: its newest frames, behind the newest frames of the committed tail
    /// when the chunk alone is too short.</summary>
    public void Stage(Tensor chunk)
    {
        ArgumentNullException.ThrowIfNull(chunk);
        if (_axis >= chunk.Shape.Rank || chunk.Shape[_axis] % _positionsPerFrame != 0 || chunk.Shape[_axis] == 0)
            throw new ArgumentException($"{chunk.Shape} does not hold whole frames of {_positionsPerFrame} positions on axis {_axis}.",
                nameof(chunk));
        int frames = checked((int)(chunk.Shape[_axis] / _positionsPerFrame));
        if (frames >= TailFrames)
        {
            _slot.Stage(StreamingTensors.Slice(_backend, chunk, _axis, (long)(frames - TailFrames) * _positionsPerFrame,
                (long)TailFrames * _positionsPerFrame));
            _stagedFrames = TailFrames;
            return;
        }

        int carried = Math.Min(TailFrames - frames, _heldFrames);
        if (carried == 0 || _slot.Current is not Tensor current)
        {
            _slot.Stage(StreamingTensors.Clone(_backend, chunk));
            _stagedFrames = frames;
            return;
        }

        using Tensor carry = StreamingTensors.Slice(_backend, current, _axis, (long)(_heldFrames - carried) * _positionsPerFrame,
            (long)carried * _positionsPerFrame);
        _slot.Stage(StreamingTensors.Concat(_backend, [carry, chunk], _axis));
        _stagedFrames = carried + frames;
    }

    /// <summary>The committed tail extended to exactly <paramref name="frames"/> frames by repeating its oldest frame in front,
    /// which is what a replicate-padded causal convolution saw before the stream had that many frames. Null before the first
    /// commit. The caller owns the result.</summary>
    public Tensor? ReplicatePadded(int frames)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(frames, 1);
        if (_slot.Current is not Tensor current)
            return null;
        if (_heldFrames >= frames)
            return StreamingTensors.Slice(_backend, current, _axis, (long)(_heldFrames - frames) * _positionsPerFrame,
                (long)frames * _positionsPerFrame);

        using Tensor oldest = StreamingTensors.Slice(_backend, current, _axis, 0, _positionsPerFrame);
        Tensor[] parts = new Tensor[frames - _heldFrames + 1];
        Array.Fill(parts, oldest);
        parts[^1] = current;
        return StreamingTensors.Concat(_backend, parts, _axis);
    }

    /// <inheritdoc />
    public void Commit()
    {
        if (!_slot.HasStaged)
            return;
        _slot.Commit();
        _heldFrames = _stagedFrames;
        _stagedFrames = 0;
    }

    /// <inheritdoc />
    public void Discard()
    {
        _slot.Discard();
        _stagedFrames = 0;
    }

    /// <inheritdoc />
    public void Reset()
    {
        _slot.Reset();
        _heldFrames = 0;
        _stagedFrames = 0;
    }

    /// <inheritdoc />
    public void Dispose() => _slot.Dispose();
}
