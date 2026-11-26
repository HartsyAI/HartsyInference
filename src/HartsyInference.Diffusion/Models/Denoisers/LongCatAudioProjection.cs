using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>The LongCat-Video-Avatar audio projection (<c>AudioProjModel</c>, <c>modules/avatar/blocks.py</c> @ 58eb336, batch 1). The first frame's
/// audio window and the latter frames' audio windows (each latent frame spans vae_scale video frames, so its window is wider) are projected
/// separately, joined along the frame axis, passed through a second ReLU projection, and expanded to <c>context_tokens</c> tokens of
/// <c>output_dim</c> per frame, then normalised with a LayerNorm over <c>output_dim</c>.</summary>
/// <remarks>
/// Input rows are one per frame: <c>audioFirst</c> is <c>[1, seqLen·blocks·channels]</c> (the first frame is always one row) and
/// <c>audioLatter</c> is <c>[f2, seqLenVf·blocks·channels]</c>, each row flattened in (window, block, channel) order. The output is
/// <c>[(1+f2)·contextTokens, outputDim]</c>, frame f's tokens in rows <c>f·contextTokens .. (f+1)·contextTokens−1</c>.
/// Weights are read under <c>audio_proj.</c> with the reference's names. All-or-nothing, like <see cref="LongCatTransformer.LoadWeights"/>.
/// </remarks>
public sealed class LongCatAudioProjection : IDisposable
{
    /// <summary>Epsilon of the output <c>nn.LayerNorm</c> (torch default).</summary>
    public const float NormEps = 1e-5f;

    private readonly int _seqLen, _seqLenVf, _blocks, _channels, _intermediate, _outputDim, _contextTokens;
    private List<Tensor> _owned = [];
    private bool _loaded;
    private Tensor? _proj1W, _proj1B, _proj1VfW, _proj1VfB, _proj2W, _proj2B, _proj3W, _proj3B, _normW, _normB;
    private int _disposed;

    /// <summary>Creates the projection geometry; <see cref="LoadWeights"/> fills it.</summary>
    public LongCatAudioProjection(int seqLen, int seqLenVf, int blocks, int channels, int intermediate, int outputDim, int contextTokens)
    {
        RequirePositive(seqLen, nameof(seqLen));
        RequirePositive(seqLenVf, nameof(seqLenVf));
        RequirePositive(blocks, nameof(blocks));
        RequirePositive(channels, nameof(channels));
        RequirePositive(intermediate, nameof(intermediate));
        RequirePositive(outputDim, nameof(outputDim));
        RequirePositive(contextTokens, nameof(contextTokens));
        _seqLen = seqLen;
        _seqLenVf = seqLenVf;
        _blocks = blocks;
        _channels = channels;
        _intermediate = intermediate;
        _outputDim = outputDim;
        _contextTokens = contextTokens;
    }

    /// <summary>Audio tokens per frame (the reference's <c>context_tokens</c>).</summary>
    public int ContextTokens => _contextTokens;

    /// <summary>Width of each output token (the reference's <c>output_dim</c>).</summary>
    public int OutputDim => _outputDim;

    /// <summary>Reads the ten tensors (four projections, each with a weight and a bias, and the output LayerNorm's weight and bias) under
    /// <paramref name="prefix"/>. The reference's <c>norm_output_audio</c> is not a config key: the avatar DiT builds the projection without
    /// it, so the output LayerNorm is always present and always loaded.</summary>
    /// <exception cref="ObjectDisposedException">The projection was disposed.</exception>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "audio_proj.")
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        ArgumentNullException.ThrowIfNull(weights);
        List<Tensor> owned = [];
        try
        {
            int inFirst = _seqLen * _blocks * _channels, inLatter = _seqLenVf * _blocks * _channels;
            Tensor p1w = Expect(weights[prefix + "proj1.weight"], [_intermediate, inFirst], prefix + "proj1.weight");
            Tensor p1b = Expect(LongCatOps.WidenF32(weights, prefix + "proj1.bias", owned), [_intermediate], prefix + "proj1.bias");
            Tensor pvw = Expect(weights[prefix + "proj1_vf.weight"], [_intermediate, inLatter], prefix + "proj1_vf.weight");
            Tensor pvb = Expect(LongCatOps.WidenF32(weights, prefix + "proj1_vf.bias", owned), [_intermediate], prefix + "proj1_vf.bias");
            Tensor p2w = Expect(weights[prefix + "proj2.weight"], [_intermediate, _intermediate], prefix + "proj2.weight");
            Tensor p2b = Expect(LongCatOps.WidenF32(weights, prefix + "proj2.bias", owned), [_intermediate], prefix + "proj2.bias");
            Tensor p3w = Expect(weights[prefix + "proj3.weight"], [_contextTokens * _outputDim, _intermediate], prefix + "proj3.weight");
            Tensor p3b = Expect(LongCatOps.WidenF32(weights, prefix + "proj3.bias", owned), [_contextTokens * _outputDim], prefix + "proj3.bias");
            Tensor nw = Expect(LongCatOps.WidenF32(weights, prefix + "norm.weight", owned), [_outputDim], prefix + "norm.weight");
            Tensor nb = Expect(LongCatOps.WidenF32(weights, prefix + "norm.bias", owned), [_outputDim], prefix + "norm.bias");

            // Every tensor loaded: release the previous set and take the new one.
            ReleaseOwned();
            _proj1W = p1w; _proj1B = p1b; _proj1VfW = pvw; _proj1VfB = pvb;
            _proj2W = p2w; _proj2B = p2b; _proj3W = p3w; _proj3B = p3b; _normW = nw; _normB = nb;
            _owned = owned;
            _loaded = true;
        }
        catch
        {
            foreach (Tensor t in owned)
                t.Dispose();
            throw;
        }
    }

    /// <summary>Projects one clip's audio windows to the per-frame context tokens. See the type remarks for the layout.</summary>
    public Tensor Forward(IBackend backend, Tensor audioFirst, Tensor audioLatter)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(audioFirst);
        ArgumentNullException.ThrowIfNull(audioLatter);
        if (!_loaded)
            throw new InvalidOperationException("The audio projection has no weights: LoadWeights was never called, or it failed.");
        int inFirst = _seqLen * _blocks * _channels, inLatter = _seqLenVf * _blocks * _channels;
        if (audioFirst.Shape.Rank != 2 || audioFirst.Shape[1] != inFirst)
            throw new ArgumentException($"The first-frame audio must be [frames, {inFirst}], got {audioFirst.Shape}.", nameof(audioFirst));
        if (audioLatter.Shape.Rank != 2 || audioLatter.Shape[1] != inLatter)
            throw new ArgumentException($"The latter-frame audio must be [frames, {inLatter}], got {audioLatter.Shape}.", nameof(audioLatter));

        int f1 = (int)audioFirst.Shape[0], f2 = (int)audioLatter.Shape[0], frames = f1 + f2;
        // Contract: callers pass exactly one first-frame row, and any other count is refused by name. The avatar DiT (not yet in src/; the
        // audio projection has no caller there yet) is to handle a one-latent-frame clip before it calls this projection.
        if (f1 != 1)
            throw new ArgumentException($"The first-frame audio must have exactly one row; it has {f1}.", nameof(audioFirst));
        // Defensive: the runtime cannot build a zero-row tensor, so this is reached only through a zero-row view. It states the reason for the
        // edge case the DiT is meant to avoid: a one-latent-frame clip has no latter frames.
        if (f2 == 0)
            throw new ArgumentException("The latter-frame audio has no rows. A one-latent-frame clip has no latter frames; the DiT must not call this with one.", nameof(audioLatter));

        // proj1 and proj1_vf, each with ReLU (LeakyRelu at slope 0 is ReLU).
        using Tensor firstLinear = new(new TensorShape(f1, _intermediate), DType.F32);
        backend.Linear(firstLinear, audioFirst, _proj1W!, _proj1B);
        using Tensor firstAct = new(firstLinear.Shape, DType.F32);
        backend.LeakyRelu(firstAct, firstLinear, 0f);

        using Tensor latterLinear = new(new TensorShape(f2, _intermediate), DType.F32);
        backend.Linear(latterLinear, audioLatter, _proj1VfW!, _proj1VfB);
        using Tensor latterAct = new(latterLinear.Shape, DType.F32);
        backend.LeakyRelu(latterAct, latterLinear, 0f);

        // Join along frames, then proj2 with ReLU.
        using Tensor joined = new(new TensorShape(frames, _intermediate), DType.F32);
        backend.Concat(joined, [firstAct, latterAct], 0);
        using Tensor second = new(joined.Shape, DType.F32);
        backend.Linear(second, joined, _proj2W!, _proj2B);
        using Tensor secondAct = new(second.Shape, DType.F32);
        backend.LeakyRelu(secondAct, second, 0f);

        // proj3 expands each frame to context_tokens · output_dim, read as one row per token, then LayerNorm over output_dim. Linear sizes
        // itself from the weight and the input, so the output takes the per-token shape directly; a Reshape would read it on the host.
        using Tensor tokens = new(new TensorShape(frames * _contextTokens, _outputDim), DType.F32);
        backend.Linear(tokens, secondAct, _proj3W!, _proj3B);
        Tensor output = new(tokens.Shape, DType.F32);
        backend.LayerNorm(output, tokens, _normW!, _normB!, NormEps);
        return output;
    }

    /// <summary>The loaded weights: matrices as stored, vectors widened; empty before a load.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _proj1W, _proj1B, _proj1VfW, _proj1VfB, _proj2W, _proj2B, _proj3W, _proj3B, _normW, _normB })
            if (t is not null)
                yield return t;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        ReleaseOwned();
        _loaded = false;
    }

    private void ReleaseOwned()
    {
        foreach (Tensor t in _owned)
            t.Dispose();
        _owned = [];
    }

    private static void RequirePositive(int value, string name)
    {
        if (value < 1)
            throw new ArgumentOutOfRangeException(name, value, $"{name} must be at least 1.");
    }

    private static Tensor Expect(Tensor tensor, long[] shape, string key)
    {
        if (tensor.Shape.Rank != shape.Length)
            throw new InvalidDataException($"{key} must have {shape.Length} dimensions, got {tensor.Shape}.");
        for (int i = 0; i < shape.Length; i++)
            if (tensor.Shape[i] != shape[i])
                throw new InvalidDataException($"{key} must be {string.Join('x', shape)}, got {tensor.Shape}.");
        return tensor;
    }
}
