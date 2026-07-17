using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection.GroundingDino;

/// <summary>DETR-style MLP prediction head (<c>GroundingDinoMLPPredictionHead</c>): <c>num_layers</c> linears with ReLU
/// between all but the last. Used for the box-regression heads (3 layers) and the decoder reference-points head
/// (2 layers).</summary>
public sealed unsafe class GroundingDinoMlp : IDisposable
{
    private readonly int _numLayers;
    private readonly Tensor?[] _w;
    private readonly Tensor?[] _b;
    private readonly int[] _outDims;
    private int _disposed;

    public GroundingDinoMlp(int numLayers, int[] outDims)
    {
        _numLayers = numLayers;
        _w = new Tensor?[numLayers];
        _b = new Tensor?[numLayers];
        _outDims = outDims;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        for (int i = 0; i < _numLayers; i++)
        {
            _w[i] = w[$"{prefix}.layers.{i}.weight"];
            _b[i] = w[$"{prefix}.layers.{i}.bias"];
        }
    }

    /// <summary>Applies the MLP to <paramref name="input"/> <c>[1, rows, inDim]</c>. Returns <c>[1, rows, outDim]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor input, int rows)
    {
        Tensor x = input;
        bool ownX = false;
        for (int i = 0; i < _numLayers; i++)
        {
            Tensor y = new(new TensorShape(1, rows, _outDims[i]), DType.F32);
            backend.Linear(y, x, _w[i]!, _b[i]);
            if (ownX) x.Dispose();
            if (i < _numLayers - 1) GdMath.Relu(y);
            x = y;
            ownX = true;
        }
        return x;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
