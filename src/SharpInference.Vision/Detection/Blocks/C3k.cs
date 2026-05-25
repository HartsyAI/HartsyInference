using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Vision.Detection.Blocks;

/// <summary>YOLO11 C3k block — structurally identical to <see cref="C2f"/> (split → chain →
/// concat-all → project) but used as the inner unit inside <see cref="C3k2"/> when its
/// <c>c3k=True</c> flag is set. Always built with <c>n=2</c> inner bottlenecks and a
/// shortcut-enabled bottleneck (matches Ultralytics' source).
/// <para>Kept as a separate class — rather than reusing <see cref="C2f"/> — so checkpoint key
/// paths stay readable: a C3k2 with c3k=True has weights at <c>m.{i}.cv1.conv.weight</c> where
/// <c>i</c> indexes the C3k sub-block, then deeper at <c>m.{i}.m.{j}.cv1.conv.weight</c> for the
/// nested bottlenecks. Reusing C2f would conflate the layers in <c>LoadWeights</c>.</para></summary>
public sealed class C3k
{
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _hiddenChannels;
    private readonly int _numBottlenecks;
    private readonly bool _shortcut;

    private readonly ConvBnSilu _cv1;
    private readonly ConvBnSilu _cv2;
    private readonly Bottleneck[] _bottlenecks;

    public C3k(int inChannels, int outChannels, int n, bool shortcut, float expansion = 0.5f)
    {
        if (n < 1)
            throw new ArgumentOutOfRangeException(nameof(n), n, "C3k requires at least one bottleneck.");
        _inChannels = inChannels;
        _outChannels = outChannels;
        _hiddenChannels = (int)(outChannels * expansion);
        _numBottlenecks = n;
        _shortcut = shortcut;

        _cv1 = new ConvBnSilu(2 * _hiddenChannels, 1, 1, 0, 0, useSilu: true);
        _cv2 = new ConvBnSilu(outChannels, 1, 1, 0, 0, useSilu: true);
        _bottlenecks = new Bottleneck[n];
        for (int i = 0; i < n; i++)
            _bottlenecks[i] = new Bottleneck(_hiddenChannels, _hiddenChannels, shortcut);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _cv1.LoadWeights(weights, $"{prefix}.cv1.conv");
        _cv2.LoadWeights(weights, $"{prefix}.cv2.conv");
        for (int i = 0; i < _bottlenecks.Length; i++)
            _bottlenecks[i].LoadWeights(weights, $"{prefix}.m.{i}");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _cv1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _cv2.EnumerateWeights()) yield return t;
        for (int i = 0; i < _bottlenecks.Length; i++)
            foreach (Tensor t in _bottlenecks[i].EnumerateWeights()) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor input)
    {
        int batch = (int)input.Shape[0];
        int height = (int)input.Shape[2];
        int width = (int)input.Shape[3];

        Tensor expanded = _cv1.Forward(backend, input);

        // Same split/chain/concat dance as C2f.
        Tensor[] accumulated = new Tensor[2 + _numBottlenecks];
        TensorShape halfShape = new TensorShape(batch, _hiddenChannels, height, width);
        accumulated[0] = new Tensor(halfShape, DType.F32);
        accumulated[1] = new Tensor(halfShape, DType.F32);
        backend.Split([accumulated[0], accumulated[1]], expanded, dim: 1);
        expanded.Dispose();

        for (int i = 0; i < _numBottlenecks; i++)
            accumulated[2 + i] = _bottlenecks[i].Forward(backend, accumulated[1 + i]);

        TensorShape concatShape = new TensorShape(batch, (2 + _numBottlenecks) * _hiddenChannels, height, width);
        Tensor concatenated = new Tensor(concatShape, DType.F32);
        backend.Concat(concatenated, accumulated, dim: 1);
        for (int i = 0; i < accumulated.Length; i++)
            accumulated[i].Dispose();

        Tensor output = _cv2.Forward(backend, concatenated);
        concatenated.Dispose();
        return output;
    }
}
