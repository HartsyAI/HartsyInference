using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using static HartsyInference.ThreeD.Models.CoLa3D.CoLa3DOps;

namespace HartsyInference.ThreeD.Models.CoLa3D;

/// <summary>Per-object mask logits: cosine similarity between projected points and projected queries, divided by a learned
/// temperature. Borrows its weights.</summary>
internal sealed class CoLa3DMaskHead
{
    private readonly int _width;
    private readonly Dense _pointProj, _queryProj;
    private readonly Tensor _temperature;

    internal CoLa3DMaskHead(IReadOnlyDictionary<string, Tensor> w, CoLa3DConfig config)
    {
        _width = config.HiddenDim;
        _pointProj = LoadDense(w, "head.pointProj");
        _queryProj = LoadDense(w, "head.queryProj");
        _temperature = w["head.temperature"];
    }

    /// <summary>Owned logits <c>[1, N, Q]</c> for points <c>[1, N, C]</c> and queries <c>[1, Q, C]</c>.</summary>
    internal Tensor Logits(IBackend backend, Tensor points, Tensor queries)
    {
        int n = (int)points.Shape[1], q = (int)queries.Shape[1];
        Tensor pp = Linear(backend, points, _pointProj, _width);
        Tensor qq = Linear(backend, queries, _queryProj, _width);
        NormalizeRows(pp.AsSpan<float>(), n, _width);
        NormalizeRows(qq.AsSpan<float>(), q, _width);

        // The query matrix [Q, C] is the weight of the similarity GEMM, so it is viewed rather than copied.
        Tensor queryMatrix = qq.Reshape(new TensorShape(q, _width));
        Tensor logits = Alloc(1, n, q);
        backend.Linear(logits, pp, queryMatrix, null);
        pp.Dispose();
        qq.Dispose();

        float temperature = _temperature.AsReadOnlySpan<float>()[0];
        Span<float> values = logits.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] /= temperature;
        return logits;
    }
}
