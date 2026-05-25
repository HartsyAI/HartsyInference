using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Vision.Detection.Blocks;

/// <summary>Spatial Pyramid Pooling — Fast. Equivalent to the older SPP block's 5/9/13 kernel
/// pyramid but ~2× faster: three sequential MaxPool2d(k=5, s=1, p=2) ops have the same effective
/// receptive field stack as a single SPP layer.
/// <list type="number">
///   <item>1×1 <c>cv1</c>: halve channels (<c>c_in → c_in / 2</c>).</item>
///   <item><c>y[0] = cv1 output</c>.</item>
///   <item><c>y[1] = MaxPool(y[0]); y[2] = MaxPool(y[1]); y[3] = MaxPool(y[2])</c>.</item>
///   <item>Concat <c>[y[0], y[1], y[2], y[3]]</c> → <c>2 * c_in</c> channels.</item>
///   <item>1×1 <c>cv2</c>: project to <c>c_out</c>.</item>
/// </list></summary>
public sealed class Sppf
{
    private readonly int _outChannels;
    private readonly int _hiddenChannels;
    private readonly int _kernel;

    private readonly ConvBnSilu _cv1;
    private readonly ConvBnSilu _cv2;

    /// <summary>Creates an SPPF block. <paramref name="kernel"/> defaults to 5 (Ultralytics default).</summary>
    public Sppf(int inChannels, int outChannels, int kernel = 5)
    {
        _outChannels = outChannels;
        _hiddenChannels = inChannels / 2;
        _kernel = kernel;

        _cv1 = new ConvBnSilu(_hiddenChannels, strideH: 1, strideW: 1, padH: 0, padW: 0, useSilu: true);
        _cv2 = new ConvBnSilu(outChannels, strideH: 1, strideW: 1, padH: 0, padW: 0, useSilu: true);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _cv1.LoadWeights(weights, $"{prefix}.cv1.conv");
        _cv2.LoadWeights(weights, $"{prefix}.cv2.conv");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _cv1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _cv2.EnumerateWeights()) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor input)
    {
        int batch = (int)input.Shape[0];
        int height = (int)input.Shape[2];
        int width = (int)input.Shape[3];

        Tensor y0 = _cv1.Forward(backend, input);

        // MaxPool with k=5, s=1, p=2 preserves spatial dims. Three sequential pools build the
        // same receptive-field pyramid the older SPP block produced with kernels 5/9/13.
        int pad = _kernel / 2;
        TensorShape ppShape = new TensorShape(batch, _hiddenChannels, height, width);
        Tensor y1 = new Tensor(ppShape, DType.F32);
        backend.MaxPool2D(y1, y0, _kernel, _kernel, 1, 1, pad, pad);
        Tensor y2 = new Tensor(ppShape, DType.F32);
        backend.MaxPool2D(y2, y1, _kernel, _kernel, 1, 1, pad, pad);
        Tensor y3 = new Tensor(ppShape, DType.F32);
        backend.MaxPool2D(y3, y2, _kernel, _kernel, 1, 1, pad, pad);

        // Concat all four pyramid levels along channel dim.
        TensorShape concatShape = new TensorShape(batch, 4 * _hiddenChannels, height, width);
        Tensor concatenated = new Tensor(concatShape, DType.F32);
        backend.Concat(concatenated, [y0, y1, y2, y3], dim: 1);
        y0.Dispose(); y1.Dispose(); y2.Dispose(); y3.Dispose();

        Tensor output = _cv2.Forward(backend, concatenated);
        concatenated.Dispose();
        return output;
    }
}
