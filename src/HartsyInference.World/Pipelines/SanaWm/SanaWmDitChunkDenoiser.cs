using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using HartsyInference.Diffusion.Streaming;

namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary><see cref="ISanaWmChunkDenoiser"/> over a loaded streaming <see cref="SanaWmDit"/>. Each guidance branch in use
/// has its own caption upload and its own <see cref="SanaWmStreamCache"/>, created on the branch's first forward; an
/// unguided stream holds only the conditional one. Borrows the DiT and the text conditioning; disposing releases the uploads
/// and the stream states. Each commit trims the backend's memory pool, so the decoder that runs next finds the memory the
/// replaced state and the passes' transients held.</summary>
public sealed class SanaWmDitChunkDenoiser : ISanaWmChunkDenoiser, IDisposable
{
    private readonly IBackend _backend;
    private readonly SanaWmDit _dit;
    private readonly SanaWmConfig _config;
    private readonly SanaWmTextConditioning _text;
    private readonly TokenWindowPolicy _softmaxWindow;
    private readonly DType? _stateDType;
    private readonly Dictionary<SanaWmGuidanceBranch, Branch> _branches = [];
    private int _disposed;

    /// <summary>Binds <paramref name="dit"/> to one stream's caption.</summary>
    /// <param name="softmaxWindow">How many earlier chunks the softmax blocks attend to.</param>
    /// <param name="stateDType">The dtype the stream's softmax windows and key tails are held in (see
    /// <see cref="SanaWmStreamState"/>); null holds them as computed.</param>
    public SanaWmDitChunkDenoiser(IBackend backend, SanaWmDit dit, SanaWmConfig config, SanaWmTextConditioning text,
        TokenWindowPolicy softmaxWindow, DType? stateDType = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(dit);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(softmaxWindow);
        _backend = backend;
        _dit = dit;
        _config = config;
        _text = text;
        _softmaxWindow = softmaxWindow;
        _stateDType = stateDType;
    }

    /// <inheritdoc/>
    public int CommittedFrames { get; private set; }

    /// <inheritdoc/>
    public Tensor Predict(Tensor latents, ReadOnlySpan<float> timesteps, SanaWmGuidanceBranch branch, ReadOnlySpan<float> cameraConditions,
        Tensor chunkPlucker, bool commit)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(latents);
        ArgumentNullException.ThrowIfNull(chunkPlucker);
        if (latents.Shape.Rank != 5 || latents.Shape[0] != 1)
            throw new ArgumentException($"A chunk forward takes latents [1, C, T, H, W]; got {latents.Shape}.", nameof(latents));

        Branch inputs = BranchFor(branch, (int)latents.Shape[3], (int)latents.Shape[4]);
        using Tensor x = Upload(latents);
        using Tensor plucker = Upload(chunkPlucker);
        using Tensor velocity = _dit.ForwardChunk(x, timesteps, inputs.Text, inputs.Mask, cameraConditions, plucker, inputs.Cache, commit);
        Tensor host = new(velocity.Shape, DType.F32, DeviceKind.Cpu);
        try
        {
            _backend.CopyTo(host, velocity);
            return host;
        }
        catch
        {
            host.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public void Commit(int latentFrames)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        foreach (Branch branch in _branches.Values)
            branch.Cache.Commit(latentFrames);
        CommittedFrames = checked(CommittedFrames + latentFrames);
        // The commit just released the state the new one replaced, and the passes before it freed their transients; both sit in
        // the allocator's pool as free blocks the driver cannot hand to anyone else. The decoder runs next and allocates from the
        // driver: on a 12 GB card the pool held about 1.9 GB of them while the decoder was refused 110 MB. Returning them costs a
        // re-map of whatever the next chunk's passes need again.
        _backend.TrimMemoryPool();
    }

    /// <inheritdoc/>
    public void Discard()
    {
        foreach (Branch branch in _branches.Values)
            branch.Cache.Discard();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (Branch branch in _branches.Values)
            branch.Dispose();
        _branches.Clear();
    }

    private Branch BranchFor(SanaWmGuidanceBranch branch, int latentHeight, int latentWidth)
    {
        if (_branches.TryGetValue(branch, out Branch? existing))
            return existing;
        // A branch that joins after the first chunk would start from an empty state while the others carry history.
        if (CommittedFrames > 0)
            throw new InvalidOperationException($"The {branch} branch was not part of the stream's first chunk; a guidance "
                + "branch cannot join a stream that is already under way.");

        Tensor text = Upload(_text.Features(branch));
        Tensor? mask = null;
        SanaWmStreamCache? cache = null;
        try
        {
            mask = Upload(_text.Mask(branch));
            cache = new SanaWmStreamCache(_backend, _config, latentHeight, latentWidth, _softmaxWindow, _stateDType);
            Branch created = new(text, mask, cache);
            _branches[branch] = created;
            return created;
        }
        catch
        {
            text.Dispose();
            mask?.Dispose();
            cache?.Dispose();
            throw;
        }
    }

    private Tensor Upload(Tensor host)
    {
        Tensor device = new(host.Shape, DType.F32, _backend.Device);
        try
        {
            _backend.CopyTo(device, host);
            return device;
        }
        catch
        {
            device.Dispose();
            throw;
        }
    }

    /// <summary>The device caption and the stream state of one guidance branch.</summary>
    private sealed class Branch(Tensor text, Tensor mask, SanaWmStreamCache cache) : IDisposable
    {
        public Tensor Text { get; } = text;

        public Tensor Mask { get; } = mask;

        public SanaWmStreamCache Cache { get; } = cache;

        public void Dispose()
        {
            Cache.Dispose();
            Text.Dispose();
            Mask.Dispose();
        }
    }
}
