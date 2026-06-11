using SharpInference.Core.Tensors;

namespace SharpInference.Interactive.Memory;

/// <summary>Bounded rolling buffer of <c>(latent, camera pose, frame index)</c> entries — the session-owned history a
/// world model conditions on (Matrix-Game's past-frame overlap + FOV-retrieved memory both read from it). The buffer
/// owns deep copies of the latents it stores (the pipeline's working latents are reused/disposed per segment) and
/// disposes them on eviction.</summary>
public sealed unsafe class FrameHistoryBuffer : IDisposable
{
    /// <summary>One stored latent frame.</summary>
    public readonly record struct Entry(Tensor Latent, float[] Pose, long FrameIndex);

    private readonly List<Entry> _entries = new();
    private readonly int _capacity;
    private int _disposed;

    public FrameHistoryBuffer(int capacity)
    {
        if (capacity <= 0) throw new ArgumentOutOfRangeException(nameof(capacity));
        _capacity = capacity;
    }

    /// <summary>Stored entry count.</summary>
    public int Count => _entries.Count;

    /// <summary>Oldest-first access.</summary>
    public Entry this[int index] => _entries[index];

    /// <summary>Copies one latent frame <c>[1, C, 1, H, W]</c> + its camera pose into the buffer, evicting (and
    /// disposing) the oldest entry when full.</summary>
    public void Add(Tensor latentFrame, ReadOnlySpan<float> pose, long frameIndex)
    {
        ThrowIfDisposed();
        if (pose.Length < 16) throw new ArgumentException("pose must hold 16 floats.", nameof(pose));
        Tensor copy = new Tensor(latentFrame.Shape, DType.F32);
        long bytes = latentFrame.Shape.ElementCount * 4;
        Buffer.MemoryCopy((float*)latentFrame.DataPointer, (float*)copy.DataPointer, bytes, bytes);
        if (_entries.Count == _capacity)
        {
            _entries[0].Latent.Dispose();
            _entries.RemoveAt(0);
        }
        _entries.Add(new Entry(copy, pose[..16].ToArray(), frameIndex));
    }

    /// <summary>The most recent <paramref name="n"/> entries, oldest-first. Throws when fewer are stored.</summary>
    public Entry[] Last(int n)
    {
        ThrowIfDisposed();
        if (n > _entries.Count) throw new ArgumentOutOfRangeException(nameof(n), $"only {_entries.Count} entries stored.");
        Entry[] result = new Entry[n];
        _entries.CopyTo(_entries.Count - n, result, 0, n);
        return result;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(FrameHistoryBuffer));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Entry e in _entries) e.Latent.Dispose();
        _entries.Clear();
    }
}
