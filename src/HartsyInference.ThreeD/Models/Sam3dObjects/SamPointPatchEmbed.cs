using System.Buffers;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// The pointmap embedder of the SAM 3D condition (upstream <c>PointPatchEmbed</c>, <c>embedder/pointmap.py</c>), inference path: the pointmap is resized
/// (nearest) to the input size, non-finite points are zeroed before the remap, every point goes through <c>point_proj</c> (invalid ones are replaced by
/// <c>invalid_xyz_token</c>, the projection's bias is not applied to them), each <c>patch x patch</c> window becomes a sequence of a pooled token plus its
/// pixels with a learned in-window position row, one timm <c>Block</c> (LayerNorm eps 1e-6, full self-attention with QKV bias, exact-GELU MLP) mixes the window,
/// and the pooled token plus the learned patch position is the window's output token. Dropout of the pointmap is the fuser's job at inference (forced
/// drop by keyword), so the training-time <c>dropped_xyz_token</c> is not read. Keys under the prefix: <c>point_proj.{weight,bias}</c>, <c>invalid_xyz_token</c>,
/// <c>pos_embed</c> (<c>[1, dim, windows, windows]</c>), <c>pos_embed_window</c>, <c>cls_token</c>, <c>blocks.0.{norm1,attn.qkv,attn.proj,norm2,mlp.fc1,mlp.fc2}.*</c>; names follow
/// the module structure and are unverified against a checkpoint. The weights are F32 and stay owned by the caller.
/// </summary>
public sealed class SamPointPatchEmbed
{
    private const float NormEps = 1e-6f;

    private readonly SamPointPatchConfig _config;
    private readonly AttentionLayer _attention;
    private readonly Tensor _projWeight, _projBias, _invalidToken, _posEmbed, _windowPos, _cls;
    private readonly Tensor _norm1Weight, _norm1Bias, _qkvWeight, _qkvBias, _attnOutWeight, _attnOutBias;
    private readonly Tensor _norm2Weight, _norm2Bias, _fc1Weight, _fc1Bias, _fc2Weight, _fc2Bias;

    public SamPointPatchEmbed(SamPointPatchConfig config, IReadOnlyDictionary<string, Tensor> weights, string prefix = "")
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(prefix);
        config.Validate();
        _config = config;
        int d = config.EmbedDim, side = config.WindowsPerSide, hidden = config.MlpHidden;
        _attention = new AttentionLayer(new AttentionLayerConfig { Heads = config.Heads, KvHeads = config.Heads, HeadDim = d / config.Heads, ComputeDType = DType.F32, AllowF16 = false });
        _projWeight = SamWeights.Require(weights, prefix + "point_proj.weight", d, 3);
        _projBias = SamWeights.Require(weights, prefix + "point_proj.bias", d);
        _invalidToken = SamWeights.Require(weights, prefix + "invalid_xyz_token", d);
        _posEmbed = SamWeights.Require(weights, prefix + "pos_embed", 1, d, side, side);
        _windowPos = SamWeights.Require(weights, prefix + "pos_embed_window", 1, config.WindowTokens, d);
        _cls = SamWeights.Require(weights, prefix + "cls_token", 1, 1, d);
        string b = prefix + "blocks.0.";
        _norm1Weight = SamWeights.Require(weights, b + "norm1.weight", d);
        _norm1Bias = SamWeights.Require(weights, b + "norm1.bias", d);
        _qkvWeight = SamWeights.Require(weights, b + "attn.qkv.weight", 3 * d, d);
        _qkvBias = SamWeights.Require(weights, b + "attn.qkv.bias", 3 * d);
        _attnOutWeight = SamWeights.Require(weights, b + "attn.proj.weight", d, d);
        _attnOutBias = SamWeights.Require(weights, b + "attn.proj.bias", d);
        _norm2Weight = SamWeights.Require(weights, b + "norm2.weight", d);
        _norm2Bias = SamWeights.Require(weights, b + "norm2.bias", d);
        _fc1Weight = SamWeights.Require(weights, b + "mlp.fc1.weight", hidden, d);
        _fc1Bias = SamWeights.Require(weights, b + "mlp.fc1.bias", hidden);
        _fc2Weight = SamWeights.Require(weights, b + "mlp.fc2.weight", d, hidden);
        _fc2Bias = SamWeights.Require(weights, b + "mlp.fc2.bias", d);
    }

    /// <summary>
    /// Embeds <paramref name="pointmap"/>, planar <c>[3, height, width]</c> (x, y, z planes; NaN or infinity marks an invalid point), into one token per window:
    /// F32 <c>[windows, dim]</c> in row-major window order, owned by the caller.
    /// </summary>
    public Tensor Embed(IBackend backend, ReadOnlySpan<float> pointmap, int height, int width)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (height <= 0 || width <= 0 || pointmap.Length != 3 * height * width)
            throw new ArgumentException($"A pointmap is 3 x {height} x {width} values; got {pointmap.Length}.", nameof(pointmap));
        int size = _config.InputSize, pixels = size * size;

        // Scratch lives in unmanaged tensor storage; only the per-pixel validity flags are managed metadata, and those come from the shared pool.
        using Tensor points = new(new TensorShape(1, pixels, 3), DType.F32);
        Span<float> xyz = points.AsSpan<float>();
        bool[] valid = ArrayPool<bool>.Shared.Rent(pixels);
        try
        {
            ResizeNearestInterleaved(pointmap, height, width, size, xyz, valid);
            SamPointRemapper.Apply(xyz, _config.Remap);
            return Project(backend, points, valid);
        }
        finally
        {
            ArrayPool<bool>.Shared.Return(valid);
        }
    }

    /// <summary>Nearest-neighbour resize of the planar map into interleaved <c>x, y, z</c> (<c>floor(dst · src / dst size)</c>, torch's rule), flagging pixels with a
    /// non-finite coordinate and zeroing them, as upstream does before the remap.</summary>
    private static void ResizeNearestInterleaved(ReadOnlySpan<float> pointmap, int height, int width, int size, Span<float> xyz, bool[] valid)
    {
        int plane = height * width;
        for (int y = 0; y < size; y++)
        {
            int sy = Math.Min(height - 1, (int)((long)y * height / size));
            for (int x = 0; x < size; x++)
            {
                int source = sy * width + Math.Min(width - 1, (int)((long)x * width / size));
                int pixel = y * size + x;
                float px = pointmap[source], py = pointmap[plane + source], pz = pointmap[2 * plane + source];
                bool finite = float.IsFinite(px) && float.IsFinite(py) && float.IsFinite(pz);
                valid[pixel] = finite;
                xyz[pixel * 3] = finite ? px : 0f;
                xyz[pixel * 3 + 1] = finite ? py : 0f;
                xyz[pixel * 3 + 2] = finite ? pz : 0f;
            }
        }
    }

    private Tensor Project(IBackend backend, Tensor points, bool[] valid)
    {
        int pixels = _config.InputSize * _config.InputSize, d = _config.EmbedDim;
        using Tensor projected = new(new TensorShape(1, pixels, d), DType.F32);
        backend.Linear(projected, points, _projWeight, _projBias);

        using Tensor windows = BuildWindows(projected, valid);
        using Tensor mixed = Block(backend, windows);
        return Pool(mixed);
    }

    /// <summary>Per window: the class token then the window's pixels, plus the in-window position rows; <c>[windows, 1 + patch², dim]</c>.</summary>
    private Tensor BuildWindows(Tensor projected, bool[] valid)
    {
        int d = _config.EmbedDim, patch = _config.PatchSize, side = _config.WindowsPerSide, size = _config.InputSize, tokens = _config.WindowTokens;
        Tensor result = new(new TensorShape(_config.Windows, tokens, d), DType.F32);
        Span<float> target = result.AsSpan<float>();
        ReadOnlySpan<float> source = projected.AsReadOnlySpan<float>();
        ReadOnlySpan<float> position = _windowPos.AsReadOnlySpan<float>();
        ReadOnlySpan<float> cls = _cls.AsReadOnlySpan<float>();
        ReadOnlySpan<float> invalid = _invalidToken.AsReadOnlySpan<float>();
        for (int wy = 0; wy < side; wy++)
        {
            for (int wx = 0; wx < side; wx++)
            {
                int window = wy * side + wx;
                Span<float> rows = target.Slice(window * tokens * d, tokens * d);
                for (int c = 0; c < d; c++)
                    rows[c] = cls[c] + position[c];
                for (int py = 0; py < patch; py++)
                {
                    for (int px = 0; px < patch; px++)
                    {
                        int pixel = (wy * patch + py) * size + wx * patch + px;
                        int token = 1 + py * patch + px;
                        ReadOnlySpan<float> value = valid[pixel] ? source.Slice(pixel * d, d) : invalid;
                        for (int c = 0; c < d; c++)
                            rows[token * d + c] = value[c] + position[token * d + c];
                    }
                }
            }
        }

        return result;
    }

    /// <summary>The timm <c>Block</c>: <c>x + attn(norm1(x))</c>, then <c>x + mlp(norm2(x))</c> with an exact-GELU MLP.</summary>
    private Tensor Block(IBackend backend, Tensor x)
    {
        int d = _config.EmbedDim, heads = _config.Heads, headDim = d / heads;
        long batch = x.Shape[0], tokens = x.Shape[1];

        using Tensor normed = new(x.Shape, DType.F32);
        backend.LayerNorm(normed, x, _norm1Weight, _norm1Bias, NormEps);
        using Tensor qkv = new(new TensorShape(batch, tokens, 3 * d), DType.F32);
        backend.Linear(qkv, normed, _qkvWeight, _qkvBias);
        Tensor attended = _attention.AttendHeadMajorOperands(
            backend,
            SamMotOps.ToHeadMajor(backend, SamMotOps.Slice(backend, qkv, 0, d), heads, headDim),
            SamMotOps.ToHeadMajor(backend, SamMotOps.Slice(backend, qkv, d, d), heads, headDim),
            SamMotOps.ToHeadMajor(backend, SamMotOps.Slice(backend, qkv, 2 * d, d), heads, headDim),
            default);
        using Tensor attendedTokens = SamMotOps.ToTokenMajor(backend, attended);
        using Tensor projected = new(x.Shape, DType.F32);
        backend.Linear(projected, attendedTokens, _attnOutWeight, _attnOutBias);
        using Tensor afterAttention = new(x.Shape, DType.F32);
        backend.Add(afterAttention, x, projected);

        using Tensor normed2 = new(x.Shape, DType.F32);
        backend.LayerNorm(normed2, afterAttention, _norm2Weight, _norm2Bias, NormEps);
        using Tensor up = new(new TensorShape(batch, tokens, _config.MlpHidden), DType.F32);
        backend.Linear(up, normed2, _fc1Weight, _fc1Bias);
        using Tensor activated = new(up.Shape, DType.F32);
        backend.GeluErf(activated, up);
        using Tensor down = new(x.Shape, DType.F32);
        backend.Linear(down, activated, _fc2Weight, _fc2Bias);
        Tensor result = new(x.Shape, DType.F32);
        backend.Add(result, afterAttention, down);
        return result;
    }

    /// <summary>The pooled (first) token of every window plus the learned patch position, <c>[windows, dim]</c>.</summary>
    private Tensor Pool(Tensor mixed)
    {
        int d = _config.EmbedDim, side = _config.WindowsPerSide, tokens = _config.WindowTokens;
        Tensor result = new(new TensorShape(_config.Windows, d), DType.F32);
        Span<float> target = result.AsSpan<float>();
        ReadOnlySpan<float> source = mixed.AsReadOnlySpan<float>();
        ReadOnlySpan<float> position = _posEmbed.AsReadOnlySpan<float>();
        int cells = side * side;
        for (int window = 0; window < _config.Windows; window++)
        {
            for (int c = 0; c < d; c++)
                target[window * d + c] = source[window * tokens * d + c] + position[c * cells + window];
        }

        return result;
    }
}
