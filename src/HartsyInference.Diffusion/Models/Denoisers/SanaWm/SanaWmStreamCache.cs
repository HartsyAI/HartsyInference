using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Streaming;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>The state one batch item of a SANA-WM chunk-causal stream carries between chunks, for every DiT block, and the
/// stream's position. Each denoising pass of a chunk reads the committed state; the pass that closes the chunk (the clean
/// pass at timestep 0) stages the next state, and <see cref="Commit"/> advances the whole DiT at once.</summary>
/// <remarks>Upstream keeps one such cache per guidance batch item; a guided stream holds one cache per branch here.</remarks>
public sealed class SanaWmStreamCache : IDisposable
{
    private readonly StreamingStateSet _states = new();
    private readonly SanaWmBlockStream[] _blocks;
    private int _disposed;

    /// <summary>Creates the empty state of a stream on a <paramref name="latentHeight"/> by <paramref name="latentWidth"/> grid.</summary>
    /// <param name="softmaxWindow">How many earlier chunks the softmax blocks attend to. Upstream's default is
    /// <c>TokenWindowPolicy.RecentChunks(2, sink: true)</c>: the first chunk and the previous one.</param>
    /// <param name="stateDType">The dtype the softmax windows and the key tails are held in between chunks: BF16 to halve them
    /// at the cost of widening them on every pass, null to hold them as computed (F32).</param>
    public SanaWmStreamCache(IBackend backend, SanaWmConfig config, int latentHeight, int latentWidth, TokenWindowPolicy softmaxWindow,
        DType? stateDType = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(softmaxWindow);
        ArgumentOutOfRangeException.ThrowIfLessThan(latentHeight, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(latentWidth, 1);
        LatentHeight = latentHeight;
        LatentWidth = latentWidth;
        _blocks = new SanaWmBlockStream[config.Depth];
        bool built = false;
        try
        {
            for (int i = 0; i < _blocks.Length; i++)
            {
                _blocks[i] = new SanaWmBlockStream(backend, _states, config.IsSoftmaxBlock(i), config.HasCameraBranch,
                    latentHeight * latentWidth, softmaxWindow, stateDType);
            }

            built = true;
        }
        finally
        {
            if (!built)
                _states.Dispose();
        }
    }

    /// <summary>Latent height the state was created for.</summary>
    public int LatentHeight { get; }

    /// <summary>Latent width the state was created for.</summary>
    public int LatentWidth { get; }

    /// <summary>Chunks committed so far.</summary>
    public int CommittedChunks => _states.CommittedChunks;

    /// <summary>Latent frames committed so far: the temporal position of the next chunk's first frame.</summary>
    public int CommittedFrames { get; private set; }

    /// <summary>Makes the state staged by a chunk of <paramref name="frames"/> latent frames current.</summary>
    public void Commit(int frames)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentOutOfRangeException.ThrowIfLessThan(frames, 1);
        _states.Commit();
        CommittedFrames = checked(CommittedFrames + frames);
    }

    /// <summary>Drops the state a failed or abandoned chunk staged; the committed state is unchanged.</summary>
    public void Discard() => _states.Discard();

    /// <summary>Returns to the state of a stream that has not started.</summary>
    public void Reset()
    {
        _states.Reset();
        CommittedFrames = 0;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _states.Dispose();
    }

    internal SanaWmBlockStream Block(int index) => _blocks[index];
}
