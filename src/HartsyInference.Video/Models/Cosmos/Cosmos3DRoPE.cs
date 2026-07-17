using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Video.Models.Cosmos;

/// <summary>Cosmos 3D rotary position embedding over the <c>(T, H, W)</c> video latent grid (research-doc §5).
/// Seeded from the Qwen-VL <c>Multimodal3DRope</c> but with the Cosmos convention: <c>head_dim/2</c> is split into
/// three <b>contiguous</b> frequency ranges (temporal, height, width), each channel drawing its rotation angle
/// from its axis's latent coordinate. The built cos/sin use the duplicated-half layout consumed by
/// <see cref="IBackend.ApplyRopeSingle"/> (Llama <c>rotate_half</c>), so the rotation reuses the existing
/// per-backend kernel — no new op. Reusable by any video-AR/diffusion model needing 3D positional rotation.
///
/// <para><b>Integration-pending numerics</b>: the exact per-axis range sizes and whether H/W share one frequency
/// schedule resolve against NVIDIA's <c>modules/embedding.py</c> at parity time (research-doc Open Q §4/§5); the
/// duplicated-half application convention is fixed.</para></summary>
public sealed unsafe class Cosmos3DRoPE
{
    private readonly int _headDim;
    private readonly int _half;
    private readonly double _theta;
    private readonly double[] _invFreq;   // [half]
    private readonly int[] _axisOfChannel;   // [half]: 0=T, 1=H, 2=W

    /// <summary>Builds the rope for the given head dim, base θ, and contiguous <c>(T,H,W)</c> partition of
    /// <c>head_dim/2</c> (must sum to <c>head_dim/2</c>).</summary>
    public Cosmos3DRoPE(int headDim, double theta, (int T, int H, int W) section)
    {
        if (headDim % 2 != 0) throw new ArgumentException($"headDim {headDim} must be even.", nameof(headDim));
        _headDim = headDim;
        _half = headDim / 2;
        _theta = theta;
        if (section.T + section.H + section.W != _half)
            throw new ArgumentException($"section {section} must sum to head_dim/2 = {_half}.", nameof(section));

        _invFreq = new double[_half];
        for (int k = 0; k < _half; k++)
            _invFreq[k] = 1.0 / Math.Pow(_theta, (double)(2 * k) / _headDim);

        _axisOfChannel = new int[_half];
        int c = 0;
        for (int k = 0; k < section.T; k++) _axisOfChannel[c++] = 0;
        for (int k = 0; k < section.H; k++) _axisOfChannel[c++] = 1;
        for (int k = 0; k < section.W; k++) _axisOfChannel[c++] = 2;
    }

    /// <summary>Total per-head dimension.</summary>
    public int HeadDim => _headDim;

    /// <summary>Builds cos/sin tables <c>[1, T·H·W, head_dim]</c> for the full latent grid, tokens in row-major
    /// <c>(t, h, w)</c> order (<c>idx = t·H·W + h·W + w</c>), in the duplicated-half layout
    /// <see cref="IBackend.ApplyRopeSingle"/> expects. Built once at prefill; a decode step slices its token row.</summary>
    public (Tensor Cos, Tensor Sin) BuildTable(int latentT, int latentH, int latentW)
    {
        int seq = latentT * latentH * latentW;
        Tensor cos = new Tensor(new TensorShape(1, seq, _headDim), DType.F32);
        Tensor sin = new Tensor(new TensorShape(1, seq, _headDim), DType.F32);
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;

        for (int t = 0; t < latentT; t++)
            for (int h = 0; h < latentH; h++)
                for (int w = 0; w < latentW; w++)
                {
                    long s = ((long)t * latentH + h) * latentW + w;
                    long baseOff = s * _headDim;
                    for (int k = 0; k < _half; k++)
                    {
                        int pos = _axisOfChannel[k] switch { 0 => t, 1 => h, _ => w };
                        double angle = pos * _invFreq[k];
                        float c = (float)Math.Cos(angle);
                        float si = (float)Math.Sin(angle);
                        cp[baseOff + k] = c; cp[baseOff + k + _half] = c;
                        sp[baseOff + k] = si; sp[baseOff + k + _half] = si;
                    }
                }
        return (cos, sin);
    }
}
