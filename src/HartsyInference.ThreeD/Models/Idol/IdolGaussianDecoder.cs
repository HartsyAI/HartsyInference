using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Models.Idol;

/// <summary>Idol's Gaussian decoder (upstream <c>GaussianFixedlenDecoder</c>) and its conversion to splats. Each anchor point is embedded and
/// passed through sixteen blocks of self-attention across the anchors, cross-attention to the 8192-token latent and an MLP; the output
/// layer gives 32 Gaussians per anchor as 480 channels (offset, colour, scale, rotation, opacity and offset scale per Gaussian).
/// <see cref="ToCloud"/> turns those channels into a <see cref="GaussianSplatCloud"/> in the layout of Idol's PLY: positions in the y-up
/// convention, log scales, logit opacities and the turned rotation quaternions. Borrows <see cref="IdolDecoderWeights"/>.</summary>
public sealed class IdolGaussianDecoder : IDisposable
{
    private const float NormEps = 1e-6f;
    private const float FinalNormEps = 1e-5f;

    private readonly IdolDecoderWeights _weights;
    private readonly IdolDecoderConfig _config;
    private readonly IdolDecoderAttention[] _self, _cross;
    private int _disposed;

    public IdolGaussianDecoder(IdolDecoderWeights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _weights = weights;
        _config = weights.Config;
        int d = _config.ModelChannels, heads = _config.Heads;
        _self = [.. Enumerable.Range(0, _config.GaussianBlocks).Select(i => new IdolDecoderAttention(weights.Tensors, $"gs.blocks.{i}.self_attn", cross: false, d, heads))];
        _cross = [.. Enumerable.Range(0, _config.GaussianBlocks).Select(i => new IdolDecoderAttention(weights.Tensors, $"gs.blocks.{i}.cross_attn", cross: true, d, heads))];
    }

    /// <summary>The per-anchor output <c>[1, A, 480]</c> for <paramref name="anchors"/> (<c>A × 3</c>, in [0, 1]) and the latent
    /// <paramref name="latent"/> <c>[1, 8192, 16]</c>. The caller owns the result.</summary>
    public Tensor Features(IBackend backend, ReadOnlySpan<float> anchors, Tensor latent)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int count = IdolAnchorEmbedding.CheckedCount(anchors), d = _config.ModelChannels;

        using Tensor points = new(new TensorShape(1, count, 3), DType.F32);
        anchors.CopyTo(points.AsSpan<float>());
        using Tensor embedding = IdolAnchorEmbedding.Build(anchors, d);
        using Tensor projected = Linear(backend, points, "gs.in_proj", d);
        using Tensor summed = new(projected.Shape, DType.F32);
        backend.Add(summed, projected, embedding);
        Tensor h = Linear(backend, summed, "gs.input_layer", d);
        try
        {
            for (int i = 0; i < _self.Length; i++)
            {
                Tensor next = Block(backend, h, i, latent);
                h.Dispose();
                h = next;
            }

            using Tensor normed = new(h.Shape, DType.F32);
            backend.LayerNormNoAffine(normed, h, FinalNormEps);
            return Linear(backend, normed, "gs.out_proj", _config.GaussianOutputChannels);
        }
        finally
        {
            h.Dispose();
        }
    }

    private Tensor Block(IBackend backend, Tensor x, int index, Tensor latent)
    {
        string p = $"gs.blocks.{index}";
        using Tensor n1 = new(x.Shape, DType.F32);
        backend.LayerNormNoAffine(n1, x, NormEps);
        using Tensor selfOut = _self[index].Forward(backend, n1, null);
        using Tensor x1 = new(x.Shape, DType.F32);
        backend.Add(x1, x, selfOut);

        using Tensor n2 = new(x.Shape, DType.F32);
        backend.LayerNorm(n2, x1, _weights[$"{p}.norm2.weight"], _weights[$"{p}.norm2.bias"], NormEps);
        using Tensor crossOut = _cross[index].Forward(backend, n2, latent);
        using Tensor x2 = new(x.Shape, DType.F32);
        backend.Add(x2, x1, crossOut);

        using Tensor n3 = new(x.Shape, DType.F32);
        backend.LayerNormNoAffine(n3, x2, NormEps);
        using Tensor hidden = Linear(backend, n3, $"{p}.mlp.mlp.0", _config.MlpHidden);
        using Tensor act = new(hidden.Shape, DType.F32);
        backend.Gelu(act, hidden);
        using Tensor down = Linear(backend, act, $"{p}.mlp.mlp.2", _config.ModelChannels);
        Tensor x3 = new(x.Shape, DType.F32);
        backend.Add(x3, x2, down);
        return x3;
    }

    /// <summary>The splats of <paramref name="features"/> <c>[1, A, 480]</c> at <paramref name="anchors"/> (<c>A × 3</c>): <c>A · 32</c> Gaussians.
    /// Opacities are written as the logit of the sigmoid clamped to [1e-6, 1 − 1e-6], as idol-sdk's PLY does, so no splat is infinite.</summary>
    public GaussianSplatCloud ToCloud(ReadOnlySpan<float> anchors, Tensor features)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        ArgumentNullException.ThrowIfNull(features);
        int per = _config.GaussiansPerAnchor;
        int count = IdolAnchorEmbedding.CheckedCount(anchors);
        if (features.DType != DType.F32 || features.Shape.Rank != 3 || features.Shape[0] != 1 || features.Shape[1] != count || features.Shape[2] != _config.GaussianOutputChannels)
            throw new ArgumentException($"The features of {count} anchors are a float32 [1, {count}, {_config.GaussianOutputChannels}] tensor from Features; got {features.DType} {features.Shape}.", nameof(features));
        long total = (long)count * per;
        float[] pos = new float[total * 3], dc = new float[total * 3], scl = new float[total * 3], rot = new float[total * 4], op = new float[total];
        ReadOnlySpan<float> f = features.AsReadOnlySpan<float>();
        ReadOnlySpan<float> perturbation = _weights["gs.points_offset_perturbation"].AsReadOnlySpan<float>();
        float baseOffsetScale = _weights["gs.base_offset_scale"].AsReadOnlySpan<float>()[0];
        float scaleOffset = GaussianSplatActivations.SoftplusInverse(_config.ScalingBias);
        float opacityOffset = MathF.Log(_config.OpacityBias / (1f - _config.OpacityBias));
        int stride = _config.GaussianOutputChannels;
        // Channel ranges per anchor, in the decoder's layout order: _xyz, _features_dc, _scaling (3 each), _rotation (4), _opacity, _offset_scale.
        int rXyz = 0, rDc = per * 3, rScl = per * 6, rRot = per * 9, rOp = per * 13, rOff = per * 14;
        float halfBox = 0.5f * _config.PerturbSize;
        for (int a = 0; a < count; a++)
        {
            ReadOnlySpan<float> row = f.Slice(a * stride, stride);
            for (int g = 0; g < per; g++)
            {
                long s = (long)a * per + g;
                float offsetScale = GaussianSplatActivations.Softplus(row[rOff + g] + baseOffsetScale);
                float x = Position(row, rXyz, perturbation, g, 0, halfBox, offsetScale, anchors[a * 3]);
                float y = Position(row, rXyz, perturbation, g, 1, halfBox, offsetScale, anchors[a * 3 + 1]);
                float z = Position(row, rXyz, perturbation, g, 2, halfBox, offsetScale, anchors[a * 3 + 2]);
                // z-up to y-up: (x, y, z) -> (x, -z, y).
                pos[s * 3] = x; pos[s * 3 + 1] = -z; pos[s * 3 + 2] = y;
                for (int c = 0; c < 3; c++)
                {
                    dc[s * 3 + c] = row[rDc + g * 3 + c] * _config.ColorScale;
                    scl[s * 3 + c] = GaussianSplatActivations.LogScale(row[rScl + g * 3 + c] * _config.ScalingScale, scaleOffset, _config.MinKernel);
                }

                float qw = row[rRot + g * 4] * _config.RotationScale + 1f, qx = row[rRot + g * 4 + 1] * _config.RotationScale;
                float qy = row[rRot + g * 4 + 2] * _config.RotationScale, qz = row[rRot + g * 4 + 3] * _config.RotationScale;
                float norm = MathF.Max(MathF.Sqrt(qw * qw + qx * qx + qy * qy + qz * qz), 1e-12f);
                (rot[s * 4], rot[s * 4 + 1], rot[s * 4 + 2], rot[s * 4 + 3]) = GaussianSplatActivations.TurnToYUp(qw / norm, qx / norm, qy / norm, qz / norm);

                float logit = row[rOp + g] * _config.OpacityScale + opacityOffset;
                float probability = Math.Clamp(1f / (1f + MathF.Exp(-logit)), 1e-6f, 1f - 1e-6f);
                op[s] = MathF.Log(probability / (1f - probability));
            }
        }

        return new GaussianSplatCloud { Positions = pos, Scales = scl, Rotations = rot, Opacities = op, ShCoefficients = dc, ShCoeffsPerSplat = 1 };
    }

    // The anchor plus the bounded offset tanh(raw·lr + perturbation) · (perturbe_size / 2) · offset_scale, shifted by the box origin -0.5.
    private float Position(ReadOnlySpan<float> row, int rXyz, ReadOnlySpan<float> perturbation, int g, int axis, float halfBox, float offsetScale, float anchor)
        => MathF.Tanh(row[rXyz + g * 3 + axis] * _config.XyzScale + perturbation[g * 3 + axis]) * halfBox * offsetScale + anchor - 0.5f;

    private Tensor Linear(IBackend backend, Tensor x, string layer, long outDim)
    {
        Tensor o = new(new TensorShape(1, x.Shape[x.Shape.Rank - 2], outDim), DType.F32);
        backend.Linear(o, x, _weights[$"{layer}.weight"], _weights[$"{layer}.bias"]);
        return o;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (IdolDecoderAttention a in _self.Concat(_cross))
            a.Dispose();
    }
}
