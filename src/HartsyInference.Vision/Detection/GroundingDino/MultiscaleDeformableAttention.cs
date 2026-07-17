using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection.GroundingDino;

/// <summary>Multi-scale deformable attention (Deformable DETR) as used by the Grounding DINO encoder self-attention
/// and decoder cross-attention. Each query predicts, per head/level/point, a 2-D sampling offset and an attention
/// weight; values are bilinearly sampled (grid_sample, align_corners=false, zero padding) from the multi-scale
/// feature maps at those locations and combined by the softmaxed weights.</summary>
public sealed unsafe class MultiscaleDeformableAttention : IDisposable
{
    private readonly int _dModel;
    private readonly int _nHeads;
    private readonly int _nLevels;
    private readonly int _nPoints;
    private Tensor? _sampOffW, _sampOffB, _attnW, _attnB, _valW, _valB, _outW, _outB;
    private int _disposed;

    public MultiscaleDeformableAttention(int dModel, int nHeads, int nLevels, int nPoints)
    {
        _dModel = dModel;
        _nHeads = nHeads;
        _nLevels = nLevels;
        _nPoints = nPoints;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _sampOffW = w[$"{prefix}.sampling_offsets.weight"]; _sampOffB = w[$"{prefix}.sampling_offsets.bias"];
        _attnW = w[$"{prefix}.attention_weights.weight"]; _attnB = w[$"{prefix}.attention_weights.bias"];
        _valW = w[$"{prefix}.value_proj.weight"]; _valB = w[$"{prefix}.value_proj.bias"];
        _outW = w[$"{prefix}.output_proj.weight"]; _outB = w[$"{prefix}.output_proj.bias"];
    }

    /// <summary>Runs deformable attention. <paramref name="queryWithPos"/> is <c>[1, Nq, d]</c> (hidden + position),
    /// <paramref name="valueSource"/> is <c>[1, Nkv, d]</c> (feature values without position). <paramref name="refPoints"/>
    /// is <c>[Nq * nLevels * coords]</c> flattened (coords 2 or 4). Returns <c>[1, Nq, d]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor queryWithPos, Tensor valueSource,
        float[] refPoints, int coords, int[][] spatialShapes, int[] levelStart)
    {
        int nq = (int)queryWithPos.Shape[1];
        int nkv = (int)valueSource.Shape[1];
        int d = _dModel, heads = _nHeads, hd = d / heads, levels = _nLevels, points = _nPoints;

        Tensor sampOff = new(new TensorShape(1, nq, heads * levels * points * 2), DType.F32);
        backend.Linear(sampOff, queryWithPos, _sampOffW!, _sampOffB);
        Tensor attn = new(new TensorShape(1, nq, heads * levels * points), DType.F32);
        backend.Linear(attn, queryWithPos, _attnW!, _attnB);
        Tensor value = new(new TensorShape(1, nkv, d), DType.F32);
        backend.Linear(value, valueSource, _valW!, _valB);

        float* offP = (float*)sampOff.DataPointer;
        float* atP = (float*)attn.DataPointer;
        float* valP = (float*)value.DataPointer;   // [nkv, heads, hd] contiguous as [nkv, d]

        // softmax attention weights over (levels*points) per (query, head)
        int lp = levels * points;
        for (int q = 0; q < nq; q++)
            for (int h = 0; h < heads; h++)
            {
                long b = ((long)q * heads + h) * lp;
                Span<float> s = new Span<float>(atP + b, lp);
                GdMath.Softmax(s);
            }

        Tensor outT = new(new TensorShape(1, nq, d), DType.F32);
        float* outP = (float*)outT.DataPointer;

        for (int q = 0; q < nq; q++)
        {
            long refBase = (long)q * levels * coords;
            for (int h = 0; h < heads; h++)
            {
                long outOff = ((long)q * heads + h) * hd;
                for (int c = 0; c < hd; c++) outP[outOff + c] = 0f;
                for (int l = 0; l < levels; l++)
                {
                    int hL = spatialShapes[l][0], wL = spatialShapes[l][1], start = levelStart[l];
                    float refX = refPoints[refBase + (long)l * coords + 0];
                    float refY = refPoints[refBase + (long)l * coords + 1];
                    for (int p = 0; p < points; p++)
                    {
                        long offIdx = ((((long)q * heads + h) * levels + l) * points + p) * 2;
                        float ox = offP[offIdx + 0], oy = offP[offIdx + 1];
                        float locX, locY;
                        if (coords == 2)
                        {
                            locX = refX + ox / wL;
                            locY = refY + oy / hL;
                        }
                        else
                        {
                            float refW = refPoints[refBase + (long)l * coords + 2];
                            float refH = refPoints[refBase + (long)l * coords + 3];
                            locX = refX + ox / points * refW * 0.5f;
                            locY = refY + oy / points * refH * 0.5f;
                        }
                        float weight = atP[(((long)q * heads + h) * levels + l) * points + p];
                        // grid_sample bilinear, align_corners=false, zero padding
                        float gx = 2f * locX - 1f, gy = 2f * locY - 1f;
                        float ix = ((gx + 1f) * wL - 1f) * 0.5f;
                        float iy = ((gy + 1f) * hL - 1f) * 0.5f;
                        int x0 = (int)MathF.Floor(ix), y0 = (int)MathF.Floor(iy);
                        int x1 = x0 + 1, y1 = y0 + 1;
                        float wx1 = ix - x0, wx0 = 1f - wx1, wy1 = iy - y0, wy0 = 1f - wy1;
                        for (int c = 0; c < hd; c++)
                        {
                            float acc = 0f;
                            acc += wy0 * wx0 * Sample(valP, start, hL, wL, heads, hd, h, x0, y0, c);
                            acc += wy0 * wx1 * Sample(valP, start, hL, wL, heads, hd, h, x1, y0, c);
                            acc += wy1 * wx0 * Sample(valP, start, hL, wL, heads, hd, h, x0, y1, c);
                            acc += wy1 * wx1 * Sample(valP, start, hL, wL, heads, hd, h, x1, y1, c);
                            outP[outOff + c] += weight * acc;
                        }
                    }
                }
            }
        }

        sampOff.Dispose();
        attn.Dispose();
        value.Dispose();

        Tensor projected = new(new TensorShape(1, nq, d), DType.F32);
        backend.Linear(projected, outT, _outW!, _outB);
        outT.Dispose();
        return projected;
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static float Sample(float* valP, int levelStart, int hL, int wL, int heads, int hd, int head, int x, int y, int c)
    {
        if (x < 0 || x >= wL || y < 0 || y >= hL) return 0f;
        long token = (long)levelStart + (long)y * wL + x;   // token index within full value sequence
        return valP[token * heads * hd + (long)head * hd + c];
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
