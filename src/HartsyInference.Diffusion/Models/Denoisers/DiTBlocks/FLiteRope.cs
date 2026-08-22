using System.Runtime.CompilerServices;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>2D Rotary Position Embedding used by F-Lite. **Non-interleaved half-rotation** (GPT-NeoX style): the head_dim is split into halves <c>[x1, x2]</c> and rotated as <c>[x1*cos + x2*sin, -x1*sin + x2*cos]</c>. This is structurally different from <see cref="FluxRope"/>'s interleaved pairwise rotation — do not substitute.
///
/// <para><b>Frequency layout</b>: the per-axis dim is <c>headDim / 2</c>. The H-axis frequencies fill the first <c>headDim/4</c> dims, the W-axis frequencies fill the next <c>headDim/4</c> dims (concat in that order). Each axis frequency table is computed as <c>inv_freq[i] = base^(-i/(headDim/2))</c> for i = 0, 2, 4 ... up to headDim/2.</para>
///
/// <para><b>Register tokens</b>: F-Lite prepends <see cref="FLiteConfig.NumRegisterTokens"/> learnable tokens before image patches. RoPE rows for the register slots are <c>cos=1, sin=0</c> — the identity rotation, so register-token Q/K is unrotated.</para></summary>
public sealed unsafe class FLiteRope
{
    private readonly int _headDim;
    private readonly int _ropeBase;
    private readonly int _maxGrid;
    private readonly float[] _invFreq;

    /// <summary>Total per-head dimension. RoPE operates on the full headDim; the half-split is internal to <see cref="ApplyRotation"/>.</summary>
    public int HeadDim => _headDim;

    /// <summary>Creates a FLiteRope with the given config. <paramref name="headDim"/> must be even (the half-split assumes 2-divisible).</summary>
    public FLiteRope(int headDim, int ropeBase = 10000, int maxGrid = 512)
    {
        if (headDim <= 0 || (headDim & 1) != 0)
            throw new ArgumentException($"headDim must be positive and even, got {headDim}.", nameof(headDim));
        _headDim = headDim;
        _ropeBase = ropeBase;
        _maxGrid = maxGrid;

        int halfDim = headDim / 2;
        int axisDim = halfDim / 2;
        _invFreq = new float[axisDim];
        for (int i = 0; i < axisDim; i++)
        {
            _invFreq[i] = (float)(1.0 / Math.Pow(ropeBase, (2.0 * i) / halfDim));
        }
    }

    /// <summary>Builds cos/sin tables for the given image patch grid plus register token padding. Output length = <c>numRegisterTokens + hPacked * wPacked</c>; output last-dim = <c>headDim/2</c> (the half each side of the split rotates against). Returned tensors are F32 with shape <c>[1, 1, T, headDim/2]</c> — broadcast-ready against multi-head Q/K of shape <c>[B, H, T, headDim]</c> (the rotation reads cos/sin against the first <c>headDim/2</c> and last <c>headDim/2</c> halves separately).</summary>
    public (Tensor cos, Tensor sin) Build(int hPacked, int wPacked, int numRegisterTokens)
    {
        if (hPacked <= 0 || wPacked <= 0)
            throw new ArgumentException("Grid dimensions must be positive.");
        if (hPacked > _maxGrid || wPacked > _maxGrid)
            throw new ArgumentException($"Grid {hPacked}×{wPacked} exceeds precomputed max {_maxGrid}×{_maxGrid}. Increase RopeMaxGrid in FLiteConfig.");

        int totalSeqLen = numRegisterTokens + hPacked * wPacked;
        int halfDim = _headDim / 2;
        int axisDim = halfDim / 2;
        TensorShape shape = new TensorShape(1, 1, totalSeqLen, halfDim);
        Tensor cosT = new Tensor(shape, DType.F32);
        Tensor sinT = new Tensor(shape, DType.F32);
        float* cosPtr = (float*)cosT.DataPointer;
        float* sinPtr = (float*)sinT.DataPointer;

        for (int r = 0; r < numRegisterTokens; r++)
        {
            for (int d = 0; d < halfDim; d++)
            {
                cosPtr[r * halfDim + d] = 1.0f;
                sinPtr[r * halfDim + d] = 0.0f;
            }
        }

        for (int hy = 0; hy < hPacked; hy++)
        {
            for (int wx = 0; wx < wPacked; wx++)
            {
                int row = numRegisterTokens + hy * wPacked + wx;
                float* cosRow = cosPtr + row * halfDim;
                float* sinRow = sinPtr + row * halfDim;
                for (int i = 0; i < axisDim; i++)
                {
                    float thetaH = hy * _invFreq[i];
                    cosRow[i] = MathF.Cos(thetaH);
                    sinRow[i] = MathF.Sin(thetaH);
                }
                for (int i = 0; i < axisDim; i++)
                {
                    float thetaW = wx * _invFreq[i];
                    cosRow[axisDim + i] = MathF.Cos(thetaW);
                    sinRow[axisDim + i] = MathF.Sin(thetaW);
                }
            }
        }

        return (cosT, sinT);
    }

    /// <summary>Builds FULL-dim cos/sin tables of shape <c>[1, T, headDim]</c> for the device <see cref="IBackend.ApplyRope"/> op (llama split-half convention). F-Lite's rotation <c>y1 = x1·c + x2·s; y2 = −x1·s + x2·c</c> equals llama's with sin negated, so the sin table is emitted NEGATED and both halves duplicate the half-dim values.</summary>
    public (Tensor cos, Tensor sin) BuildDeviceTables(int hPacked, int wPacked, int numRegisterTokens)
    {
        (Tensor cosHalf, Tensor sinHalf) = Build(hPacked, wPacked, numRegisterTokens);
        int totalSeqLen = (int)cosHalf.Shape[2];
        int halfDim = _headDim / 2;
        Tensor cosFull = new Tensor(new TensorShape(1, totalSeqLen, _headDim), DType.F32);
        Tensor sinFull = new Tensor(new TensorShape(1, totalSeqLen, _headDim), DType.F32);
        float* ch = (float*)cosHalf.DataPointer;
        float* sh = (float*)sinHalf.DataPointer;
        float* cf = (float*)cosFull.DataPointer;
        float* sf = (float*)sinFull.DataPointer;
        for (int t = 0; t < totalSeqLen; t++)
        {
            for (int i = 0; i < halfDim; i++)
            {
                float c = ch[(long)t * halfDim + i];
                float s = sh[(long)t * halfDim + i];
                cf[(long)t * _headDim + i] = c;
                cf[(long)t * _headDim + halfDim + i] = c;
                sf[(long)t * _headDim + i] = -s;
                sf[(long)t * _headDim + halfDim + i] = -s;
            }
        }
        cosHalf.Dispose();
        sinHalf.Dispose();
        return (cosFull, sinFull);
    }

    /// <summary>Applies the F-Lite half-rotation in place to a multi-head tensor of shape <c>[B, H, T, headDim]</c>. Reads cos/sin of shape <c>[1, 1, T, headDim/2]</c>. Caller is responsible for the matching head-major layout.</summary>
    public void ApplyRotation(Tensor x, Tensor cos, Tensor sin, int batch, int numHeads, int seqLen)
    {
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new ArgumentException("FLiteRope.ApplyRotation expects F32 tensors.");
        if (x.Shape.Rank != 4 || x.Shape[3] != _headDim)
            throw new ArgumentException($"x must be [B, H, T, {_headDim}] (head_dim={_headDim}); got {x.Shape}.");

        int halfDim = _headDim / 2;
        float* xPtr = (float*)x.DataPointer;
        float* cosPtr = (float*)cos.DataPointer;
        float* sinPtr = (float*)sin.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    long baseIdx = ((long)b * numHeads + h) * seqLen * _headDim + (long)t * _headDim;
                    long rotIdx = (long)t * halfDim;
                    RotateOnePosition(xPtr + baseIdx, cosPtr + rotIdx, sinPtr + rotIdx, halfDim);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RotateOnePosition(float* x, float* cos, float* sin, int halfDim)
    {
        // y1 = x1 * cos + x2 * sin
        // y2 = -x1 * sin + x2 * cos
        // Compute into a stack temp because we'd otherwise overwrite x1 before reading it for y2.
        for (int i = 0; i < halfDim; i++)
        {
            float x1 = x[i];
            float x2 = x[halfDim + i];
            float c = cos[i];
            float s = sin[i];
            x[i] = x1 * c + x2 * s;
            x[halfDim + i] = -x1 * s + x2 * c;
        }
    }
}
