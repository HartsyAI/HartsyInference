using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Multimodal;

namespace HartsyInference.LLM.Kimi.Vision;

/// <summary>The Kimi K3.1 MoonViT-V2 tower with its patch merger, as an <see cref="IVlmImageEncoder"/>. Takes one normalized F32 image
/// <c>[1, 3, H, W]</c> whose sides are multiples of <c>patch·merge</c>, and returns <c>[1, (H/patch/merge)·(W/patch/merge), TextHidden]</c>.</summary>
public sealed unsafe class MoonViTEncoder : IVlmImageEncoder
{
    private readonly KimiVisionConfig _config;
    private readonly KimiVisionWeights _weights;
    private int _disposed;

    /// <summary>Takes ownership of <paramref name="weights"/>, which must hold one block per configured layer.</summary>
    public MoonViTEncoder(KimiVisionConfig config, KimiVisionWeights weights)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        config.Validate();
        if (weights.Blocks.Count != config.Layers)
            throw new ArgumentException($"The weights hold {weights.Blocks.Count} blocks; the config has {config.Layers}.", nameof(weights));
        _config = config;
        _weights = weights;
        int side = config.ImageSize / (config.PatchSize * config.MergeSize);
        TokensPerImage = side * side;
    }

    /// <summary>Tokens for <see cref="ImageSize"/>; other sizes emit a count set by their grid.</summary>
    public int TokensPerImage { get; }

    /// <inheritdoc/>
    public int ImageSize => _config.ImageSize;

    /// <inheritdoc/>
    public float[] ImageMean { get; } = [0.5f, 0.5f, 0.5f];

    /// <inheritdoc/>
    public float[] ImageStd { get; } = [0.5f, 0.5f, 0.5f];

    /// <inheritdoc/>
    public string Family => "kimik3";

    /// <inheritdoc/>
    public Tensor Encode(IBackend backend, Tensor pixelValues)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(pixelValues);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        if (pixelValues.DType != DType.F32 || pixelValues.Shape.Rank != 4 || pixelValues.Shape[0] != 1 || pixelValues.Shape[1] != 3)
            throw new ArgumentException($"The Kimi vision tower takes one F32 image [1, 3, H, W]; got {pixelValues.DType} {pixelValues.Shape}.", nameof(pixelValues));

        int factor = _config.PatchSize * _config.MergeSize;
        int height = (int)pixelValues.Shape[2], width = (int)pixelValues.Shape[3];
        if (height == 0 || width == 0 || height % factor != 0 || width % factor != 0)
            throw new ArgumentException($"Image sides must be multiples of {factor}; got {width}x{height}.", nameof(pixelValues));
        int gh = height / _config.PatchSize, gw = width / _config.PatchSize, np = gh * gw;
        int hidden = _config.Hidden;

        Tensor patches = VisionPatchOps.Patchify(backend, pixelValues, gh, gw, _config.MergeSize, _config.PatchSize);
        Tensor embed = new(new TensorShape(np, hidden), DType.F32);
        backend.Linear(embed, patches, _weights.PatchEmbed, null);
        patches.Dispose();

        Tensor pos = new(new TensorShape(np, hidden), DType.F32);
        backend.Sync();
        KimiVisionPosEmbed.Resample(pos.AsSpan<float>(), _weights.PosEmbed.AsReadOnlySpan<float>(), _config.InitPosGrid, hidden, gh, gw,
            _config.MergeSize);
        Tensor x = new(new TensorShape(np, hidden), DType.F32);
        backend.Add(x, embed, pos);
        embed.Dispose();
        pos.Dispose();

        (float[] cos, float[] sin) = KimiVisionRope.Build(gh, gw, _config.HeadDim, _config.MergeSize, _config.RopeTheta);
        int pairs = _config.HeadDim / 2;
        using Tensor cosT = Table(cos, np, pairs);
        using Tensor sinT = Table(sin, np, pairs);

        foreach (KimiVisionBlockWeights block in _weights.Blocks)
        {
            Tensor next = Block(backend, x, block, cosT, sinT, np);
            x.Dispose();
            x = next;
        }

        TensorShape batched = new(1, np, hidden);
        Tensor normed = new(batched, DType.F32);
        backend.RmsNorm(normed, x.Reshape(batched), _weights.FinalNorm, _config.BlockNormEpsilon);
        x.Dispose();
        Tensor merged = KimiVisionProjector.Apply(backend, normed.Reshape(new TensorShape(np, hidden)), _weights.ProjectorUp,
            _weights.ProjectorDown, _weights.ProjectorNorm, _config.MergeSize, _config.ProjectorEpsilon);
        normed.Dispose();
        backend.Sync();
        return merged;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _weights.Dispose();
    }

    private Tensor Block(IBackend backend, Tensor x, KimiVisionBlockWeights b, Tensor cos, Tensor sin, int np)
    {
        int hidden = _config.Hidden, heads = _config.Heads, hd = _config.HeadDim, mlp = _config.MlpDim;
        float eps = _config.BlockNormEpsilon;
        TensorShape flat = new(1, np, hidden);
        TensorShape headed = new(1, np, heads, hd);

        Tensor normed = new(flat, DType.F32);
        backend.RmsNorm(normed, x.Reshape(flat), b.Norm0, eps);
        Tensor q = new(headed, DType.F32), k = new(headed, DType.F32), v = new(headed, DType.F32);
        backend.Linear(q, normed, b.Q, null);
        backend.Linear(k, normed, b.K, null);
        backend.Linear(v, normed, b.V, null);
        normed.Dispose();
        backend.Rotary(RotaryCall.TokenMajor(q, cos, sin, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPrefix));
        backend.Rotary(RotaryCall.TokenMajor(k, cos, sin, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPrefix));

        Tensor qh = new(new TensorShape(1, heads, np, hd), DType.F32);
        Tensor kh = new(new TensorShape(1, heads, np, hd), DType.F32);
        Tensor vh = new(new TensorShape(1, heads, np, hd), DType.F32);
        backend.Permute0213(qh, q, np, heads, hd);
        backend.Permute0213(kh, k, np, heads, hd);
        backend.Permute0213(vh, v, np, heads, hd);
        q.Dispose();
        k.Dispose();
        v.Dispose();

        Tensor attn = new(new TensorShape(1, heads, np, hd), DType.F32);
        backend.ScaledDotProductAttention(attn, qh, kh, vh, null, 1f / MathF.Sqrt(hd));
        qh.Dispose();
        kh.Dispose();
        vh.Dispose();

        Tensor attnFlat = new(flat, DType.F32);
        backend.Permute0213(attnFlat, attn, heads, np, hd);
        attn.Dispose();
        Tensor attnOut = new(flat, DType.F32);
        backend.Linear(attnOut, attnFlat, b.Out, null);
        attnFlat.Dispose();
        Tensor after = new(flat, DType.F32);
        backend.Add(after, x.Reshape(flat), attnOut);
        attnOut.Dispose();

        Tensor normed1 = new(flat, DType.F32);
        backend.RmsNorm(normed1, after, b.Norm1, eps);
        Tensor up = new(new TensorShape(1, np, mlp), DType.F32);
        backend.Linear(up, normed1, b.Fc0, null);
        normed1.Dispose();
        Tensor act = new(new TensorShape(1, np, mlp), DType.F32);
        backend.Gelu(act, up);
        up.Dispose();
        Tensor down = new(flat, DType.F32);
        backend.Linear(down, act, b.Fc1, null);
        act.Dispose();
        Tensor result = new(flat, DType.F32);
        backend.Add(result, after, down);
        after.Dispose();
        down.Dispose();
        return result;
    }

    private static Tensor Table(float[] values, int rows, int cols)
    {
        Tensor t = new(new TensorShape(rows, cols), DType.F32);
        values.AsSpan().CopyTo(t.AsSpan<float>());
        return t;
    }
}
