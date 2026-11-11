using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>diffusers' <c>Transformer2DModel</c> (linear projections): GroupNorm, <c>proj_in</c>, the 2.5D transformer block, <c>proj_out</c> and the
/// residual, applied to every material's <c>[views, channels, height, width]</c> activation.</summary>
internal sealed class HunyuanPaintSpatialTransformer
{
    private readonly int _channels, _groups;
    private readonly HunyuanPaintBlock _block;
    private Tensor? _normW, _normB, _projInW, _projInB, _projOutW, _projOutB;

    internal HunyuanPaintSpatialTransformer(HunyuanPaintConfig config, int channels, int heads, bool extended, string blockName)
    {
        _channels = channels;
        _groups = config.NormGroups;
        _block = new HunyuanPaintBlock(config, channels, heads, extended, blockName);
    }

    internal void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _normW = w[$"{prefix}.norm.weight"];
        _normB = w[$"{prefix}.norm.bias"];
        _projInW = Matrix(w[$"{prefix}.proj_in.weight"]);
        _projInB = w[$"{prefix}.proj_in.bias"];
        _projOutW = Matrix(w[$"{prefix}.proj_out.weight"]);
        _projOutB = w[$"{prefix}.proj_out.bias"];
        _block.LoadWeights(w, $"{prefix}.transformer_blocks.0");
    }

    internal IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _normW, _normB, _projInW, _projInB, _projOutW, _projOutB })
            if (t is not null)
                yield return t;
        foreach (Tensor t in _block.EnumerateWeights())
            yield return t;
    }

    /// <summary>Runs the transformer over every material; <paramref name="x"/> is borrowed and new tensors are returned.</summary>
    internal Tensor[] Forward(IBackend backend, Tensor[] x, HunyuanPaintBlockState state)
    {
        int views = (int)x[0].Shape[0], height = (int)x[0].Shape[2], width = (int)x[0].Shape[3], spatial = height * width;
        TensorShape grid = new(views, _channels, height, width), tokens = new(views, spatial, _channels);
        Tensor[] hidden = new Tensor[x.Length];
        for (int m = 0; m < x.Length; m++)
        {
            Tensor normed = new(grid, DType.F32);
            backend.GroupNorm(normed, x[m], _normW!, _normB!, _groups, 1e-6f);
            Tensor transposed = new(tokens, DType.F32);
            backend.Transpose2D(transposed, normed, _channels, spatial);
            normed.Dispose();
            hidden[m] = new Tensor(tokens, DType.F32);
            backend.Linear(hidden[m], transposed, _projInW!, _projInB);
            transposed.Dispose();
        }
        hidden = _block.Forward(backend, hidden, state);
        Tensor[] output = new Tensor[x.Length];
        for (int m = 0; m < x.Length; m++)
        {
            Tensor projected = new(tokens, DType.F32);
            backend.Linear(projected, hidden[m], _projOutW!, _projOutB);
            hidden[m].Dispose();
            Tensor image = new(grid, DType.F32);
            backend.Transpose2D(image, projected, spatial, _channels);
            projected.Dispose();
            output[m] = new Tensor(grid, DType.F32);
            backend.Add(output[m], image, x[m]);
            image.Dispose();
        }
        return output;
    }

    private static Tensor Matrix(Tensor weight) => weight.Shape.Rank == 2 ? weight : weight.Reshape(new TensorShape(weight.Shape[0], weight.Shape[1]));
}
