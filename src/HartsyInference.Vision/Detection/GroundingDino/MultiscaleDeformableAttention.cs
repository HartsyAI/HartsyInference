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
        int d = _dModel, heads = _nHeads, levels = _nLevels, points = _nPoints;

        Tensor sampOff = new(new TensorShape(1, nq, heads * levels * points * 2), DType.F32);
        backend.Linear(sampOff, queryWithPos, _sampOffW!, _sampOffB);
        Tensor attn = new(new TensorShape(1, nq, heads * levels * points), DType.F32);
        backend.Linear(attn, queryWithPos, _attnW!, _attnB);
        Tensor value = new(new TensorShape(1, nkv, d), DType.F32);
        backend.Linear(value, valueSource, _valW!, _valB);

        // Flatten spatial shapes to [levels*2] (h,w) for the backend op; ref points → a device-uploadable tensor.
        Span<int> shapesFlat = stackalloc int[levels * 2];
        for (int l = 0; l < levels; l++) { shapesFlat[l * 2] = spatialShapes[l][0]; shapesFlat[l * 2 + 1] = spatialShapes[l][1]; }
        Tensor refT = new(new TensorShape(refPoints.Length), DType.F32);
        refPoints.AsSpan().CopyTo(new Span<float>((void*)refT.DataPointer, refPoints.Length));

        // GDINO stores per-level reference points: stride levels*coords per query, coords per level.
        Tensor outT = new(new TensorShape(1, nq, d), DType.F32);
        backend.DeformableAttention(outT, value, sampOff, attn, refT, shapesFlat, levelStart,
            heads, levels, points, coords, levels * coords, coords);

        sampOff.Dispose();
        attn.Dispose();
        value.Dispose();
        refT.Dispose();

        Tensor projected = new(new TensorShape(1, nq, d), DType.F32);
        backend.Linear(projected, outT, _outW!, _outB);
        outT.Dispose();
        return projected;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
