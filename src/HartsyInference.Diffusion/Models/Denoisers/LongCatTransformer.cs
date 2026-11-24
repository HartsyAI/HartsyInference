using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>LongCat-Video DiT (<c>LongCatVideoTransformer3DModel</c>, meituan-longcat/LongCat-Video @ 58eb336): Conv3d patch
/// embedding, a per-frame timestep MLP,
/// the caption MLP, <see cref="LongCatConfig.Depth"/> <see cref="LongCatBlock"/>s and a per-frame adaLN final layer. B=1. The
/// audio cross-attention of the avatar
/// models belongs to <see cref="LongCatBlock"/> in the phase that adds it; the forward here is the shape it extends.</summary>
public sealed unsafe class LongCatTransformer : IDisposable, IStreamableDenoiser
{
    private readonly LongCatConfig _config;
    private readonly LongCatBlock[] _blocks;
    private readonly WanRope _rope;
    private readonly WanForwardCaches _caches = new();
    private readonly Dictionary<(int Frames, int TokensPerFrame), Tensor> _frameIndexes = [];
    private int _disposed;

    private Tensor? _patchW2d, _patchBias;
    private Tensor? _time1W, _time1B, _time2W, _time2B;
    private Tensor? _text1W, _text1B, _text2W, _text2B;
    private Tensor? _finalAdaW, _finalAdaB, _finalW, _finalB;

    /// <summary>Creates the transformer; <see cref="LoadWeights"/> fills it.</summary>
    public LongCatTransformer(LongCatConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        _config = config;
        _blocks = new LongCatBlock[config.Depth];
        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i] = new LongCatBlock(config);
        _rope = new WanRope(config.HeadDim, LongCatConfig.RopeTheta, 0);
    }

    /// <summary>The geometry.</summary>
    public LongCatConfig Config => _config;

    /// <summary>Receives named intermediate stages (<c>x_embedder</c>, <c>t_embedder</c>, <c>y_embedder</c>, <c>rope_cos</c>,
    /// <c>blockN.*</c>, <c>final_layer</c>, <c>velocity</c>) when set.</summary>
    public Action<string, Tensor>? Probe { get; set; }

    /// <inheritdoc/>
    public int BlockCount => _blocks.Length;

    /// <inheritdoc/>
    public IStreamingBlock GetBlock(int idx) => _blocks[idx];

    /// <inheritdoc/>
    public Action<int>? BeforeBlockForward { get; set; }

    /// <summary>Reads every tensor from the checkpoint dictionary (reference key names).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _patchW2d = WanDitOps.Reshape2d(w["x_embedder.proj.weight"], _config.HiddenSize, _config.PatchVector);
        _patchBias = TensorCasts.LoadF32(w, "x_embedder.proj.bias");
        _time1W = w["t_embedder.mlp.0.weight"];
        _time1B = w["t_embedder.mlp.0.bias"];
        _time2W = w["t_embedder.mlp.2.weight"];
        _time2B = w["t_embedder.mlp.2.bias"];
        _text1W = w["y_embedder.y_proj.0.weight"];
        _text1B = w["y_embedder.y_proj.0.bias"];
        _text2W = w["y_embedder.y_proj.2.weight"];
        _text2B = w["y_embedder.y_proj.2.bias"];
        _finalAdaW = w["final_layer.adaLN_modulation.1.weight"];
        _finalAdaB = w["final_layer.adaLN_modulation.1.bias"];
        _finalW = w["final_layer.linear.weight"];
        _finalB = w["final_layer.linear.bias"];
        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i].LoadWeights(w, $"blocks.{i}");
    }

    /// <summary>Every weight: the shared ones, then each block's.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in EnumerateSharedWeights())
            yield return t;
        foreach (LongCatBlock block in _blocks)
            foreach (Tensor t in block.EnumerateWeights())
                yield return t;
    }

    /// <inheritdoc/>
    public IEnumerable<Tensor> EnumerateSharedWeights()
    {
        foreach (Tensor? t in new[] { _patchW2d, _patchBias, _time1W, _time1B, _time2W, _time2B, _text1W, _text1B, _text2W, _text2B,
            _finalAdaW, _finalAdaB, _finalW, _finalB })
        {
            if (t is not null)
                yield return t;
        }
    }

    /// <summary>Velocity prediction (the model output, before the sampler's sign flip). <paramref name="latent"/> is <c>[1, z, T, H, W]</c>;
    /// <paramref name="encoder"/> the umT5 features <c>[L, captionChannels]</c>; <paramref name="frameTimesteps"/> one timestep
    /// for all frames or one per
    /// latent frame (<see cref="LongCatConditioning.FrameTimesteps"/>); <paramref name="validTextTokens"/> the unpadded caption length, default all.
    /// Rows past it are zeroed after the caption projection and still attended when <see cref="LongCatConfig.TextTokensZeroPad"/>
    /// is set, else dropped.
    /// Returns <c>[1, z, T, H, W]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor encoder, ReadOnlySpan<float> frameTimesteps, in LongCatConditioning conditioning,
        int validTextTokens = -1)
    {
        if (latent.Shape.Rank != 5 || latent.Shape[0] != 1 || latent.Shape[1] != _config.InChannels)
            throw new ArgumentException($"The latent is [1, {_config.InChannels}, T, H, W]; got {latent.Shape}.", nameof(latent));
        int frames = (int)latent.Shape[2], height = (int)latent.Shape[3], width = (int)latent.Shape[4];
        (int pt, int ph, int pw) = _config.PatchSize;
        if (height % ph != 0 || width % pw != 0 || frames % pt != 0)
            throw new ArgumentException($"The latent {latent.Shape} does not tile into {_config.PatchSize} patches.", nameof(latent));
        int gridH = height / ph, gridW = width / pw, hw = gridH * gridW, dim = _config.HiddenSize;
        if (frameTimesteps.Length != 1 && frameTimesteps.Length != frames)
            throw new ArgumentException(
                $"Give one timestep or one per latent frame ({frames}); got {frameTimesteps.Length}.", nameof(frameTimesteps));
        conditioning.Validate(frames);

        Tensor hidden = WanDitOps.Patchify(backend, latent, _config.InChannels, dim, _config.PatchSize, _patchW2d!, _patchBias);
        Probe?.Invoke("x_embedder", hidden);

        using Tensor timeEmbed = TimestepEmbedding(backend, frameTimesteps, frames);
        Probe?.Invoke("t_embedder", timeEmbed);
        using Tensor text = CaptionEmbedding(backend, encoder, validTextTokens);
        Probe?.Invoke("y_embedder", text);

        int roperef = conditioning.NumRefLatents > 0 ? conditioning.RefImgIndex + 1 : 0;
        LongCatConditioning ropeConditioning = conditioning;
        (Tensor cos, Tensor sin) = _caches.RopeCosSin((frames, gridH, gridW, roperef), () => BuildRope(frames, gridH, gridW, ropeConditioning));
        if (Probe is not null)
        {
            Probe("rope_cos", cos);
            Probe("rope_sin", sin);
        }
        AttentionSegment[] segments = conditioning.AttentionPlan(hw, frames);
        LongCatBlockInputs inputs = new(frames, hw, conditioning.NumCondLatents, cos, sin, segments, FrameRowIndex(frames, hw));

        Tensor cur = hidden;
        for (int i = 0; i < _blocks.Length; i++)
        {
            BeforeBlockForward?.Invoke(i);
            Tensor next = _blocks[i].Forward(backend, cur, text, timeEmbed, in inputs, Probe, Probe is null ? "" : $"block{i}");
            cur.Dispose();
            cur = next;
        }

        using Tensor finalModulation = LongCatOps.AdaLnProjection(backend, timeEmbed, _finalAdaW!, _finalAdaB);
        using Tensor modulated = LongCatOps.NormModulate(backend, cur, finalModulation, dim, 0, hw);
        cur.Dispose();
        using Tensor projected = new(new TensorShape((long)frames * hw, _finalW!.Shape[0]), DType.F32);
        backend.Linear(projected, modulated, _finalW!, _finalB);
        Probe?.Invoke("final_layer", projected);
        Tensor velocity = WanDitOps.Unpatchify(projected, _config.OutChannels, frames / pt, gridH, gridW, _config.PatchSize);
        Probe?.Invoke("velocity", velocity);
        return velocity;
    }

    /// <summary>Sinusoidal features (cos then sin, base 10000) of each frame's timestep, then Linear, SiLU, Linear: <c>[frames, adaln]</c>.</summary>
    private Tensor TimestepEmbedding(IBackend backend, ReadOnlySpan<float> timesteps, int frames)
    {
        int features = _config.FrequencyEmbeddingSize;
        using Tensor sinusoid = new(new TensorShape(frames, features), DType.F32);
        for (int f = 0; f < frames; f++)
        {
            using Tensor row = sinusoid.SliceRows(f, 1);
            DiTUtils.SinusoidalTimestepEmbedding(row, timesteps.Length == 1 ? timesteps[0] : timesteps[f], 1, features, 10000f);
        }
        Probe?.Invoke("t_freq", sinusoid);
        using Tensor first = new(new TensorShape(frames, _config.AdalnTembedDim), DType.F32);
        backend.Linear(first, sinusoid, _time1W!, _time1B);
        using Tensor activated = new(first.Shape, DType.F32);
        backend.Silu(activated, first);
        Tensor embedding = new(first.Shape, DType.F32);
        backend.Linear(embedding, activated, _time2W!, _time2B);
        return embedding;
    }

    private Tensor CaptionEmbedding(IBackend backend, Tensor encoder, int validTextTokens)
    {
        int dim = _config.HiddenSize;
        Tensor projected = _caches.TextProj(backend, encoder, dim, _text1W!, _text1B, _text2W!, _text2B);
        int rows = (int)projected.Shape[0];
        int valid = validTextTokens < 0 ? rows : validTextTokens;
        if (valid < 1 || valid > rows)
            throw new ArgumentOutOfRangeException(nameof(validTextTokens), validTextTokens, $"The valid caption length must lie in [1, {rows}].");
        if (valid == rows)
            return LongCatOps.Copy(backend, projected);

        using Tensor kept = new(new TensorShape(valid, dim), DType.F32);
        backend.SliceRows(kept, projected, 0);
        if (!_config.TextTokensZeroPad)
            return LongCatOps.Copy(backend, kept);
        Tensor padded = new(new TensorShape(rows, dim), DType.F32);
        using Tensor zeros = LongCatOps.Zeros(new TensorShape(rows - valid, dim));
        backend.Concat(padded, [kept, zeros], 0);
        return padded;
    }

    private (Tensor Cos, Tensor Sin) BuildRope(int frames, int gridH, int gridW, in LongCatConditioning conditioning)
    {
        int[] indices = new int[frames];
        if (conditioning.NumRefLatents > 0)
        {
            indices[0] = conditioning.RefImgIndex;
            for (int i = 1; i < frames; i++)
                indices[i] = i - 1;
        }
        else
        {
            for (int i = 0; i < frames; i++)
                indices[i] = i;
        }
        return _rope.BuildCosSin(indices, gridH, gridW);
    }

    private Tensor FrameRowIndex(int frames, int tokensPerFrame)
    {
        if (!_frameIndexes.TryGetValue((frames, tokensPerFrame), out Tensor? index))
        {
            if (_frameIndexes.Count >= 4)
            {
                foreach (Tensor old in _frameIndexes.Values)
                    old.Dispose();
                _frameIndexes.Clear();
            }
            index = LongCatOps.FrameRowIndex(frames, tokensPerFrame);
            _frameIndexes[(frames, tokensPerFrame)] = index;
        }
        return index;
    }

    /// <summary>Drops the references; the weight tensors belong to the checkpoint that supplied them.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            foreach (Tensor index in _frameIndexes.Values)
                index.Dispose();
            _frameIndexes.Clear();
            _patchW2d = _patchBias = _time1W = _time1B = _time2W = _time2B = _text1W = _text1B = _text2W = _text2B = null;
            _finalAdaW = _finalAdaB = _finalW = _finalB = null;
        }
        GC.SuppressFinalize(this);
    }
}
