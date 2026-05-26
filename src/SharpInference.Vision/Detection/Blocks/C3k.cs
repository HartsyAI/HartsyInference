using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Vision.Detection.Blocks;

/// <summary>YOLO11 C3k block — the "CSP Bottleneck with 3 convolutions" variant from older YOLO
/// generations, reused as the inner unit of <see cref="C3k2"/> when its <c>c3k=True</c> flag is
/// set. The structure is <i>not</i> a C2f variant — it has three projection convs:
/// <list type="bullet">
///   <item><c>cv1</c>: 1×1 left-branch projection (<c>c_in → c_</c>).</item>
///   <item><c>cv2</c>: 1×1 right-branch projection in parallel (<c>c_in → c_</c>).</item>
///   <item><c>m</c>: sequential chain of <c>n</c> <see cref="Bottleneck"/> instances applied to the cv1 output.</item>
///   <item><c>cv3</c>: 1×1 projection of <c>concat([m(cv1(x)), cv2(x)])</c> back to <c>c_out</c>.</item>
/// </list>
/// <para>Critically — the bottlenecks are applied SEQUENTIALLY (a chain) and only the last
/// output reaches the concat, unlike C2f's "concat every intermediate" pattern.</para></summary>
public sealed class C3k
{
    private readonly int _hiddenChannels;
    private readonly int _numBottlenecks;

    private readonly ConvBnSilu _cv1;
    private readonly ConvBnSilu _cv2;
    private readonly ConvBnSilu _cv3;
    private readonly Bottleneck[] _bottlenecks;

    public C3k(int inChannels, int outChannels, int n, bool shortcut, float expansion = 0.5f)
    {
        if (n < 1)
            throw new ArgumentOutOfRangeException(nameof(n), n, "C3k requires at least one bottleneck.");
        _hiddenChannels = (int)(outChannels * expansion);
        _numBottlenecks = n;

        _cv1 = new ConvBnSilu(_hiddenChannels, 1, 1, 0, 0, useSilu: true);
        _cv2 = new ConvBnSilu(_hiddenChannels, 1, 1, 0, 0, useSilu: true);
        _cv3 = new ConvBnSilu(outChannels, 1, 1, 0, 0, useSilu: true);
        _bottlenecks = new Bottleneck[n];
        for (int i = 0; i < n; i++)
            _bottlenecks[i] = new Bottleneck(_hiddenChannels, _hiddenChannels, shortcut);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _cv1.LoadWeights(weights, $"{prefix}.cv1.conv");
        _cv2.LoadWeights(weights, $"{prefix}.cv2.conv");
        _cv3.LoadWeights(weights, $"{prefix}.cv3.conv");
        for (int i = 0; i < _bottlenecks.Length; i++)
            _bottlenecks[i].LoadWeights(weights, $"{prefix}.m.{i}");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _cv1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _cv2.EnumerateWeights()) yield return t;
        foreach (Tensor t in _cv3.EnumerateWeights()) yield return t;
        for (int i = 0; i < _bottlenecks.Length; i++)
            foreach (Tensor t in _bottlenecks[i].EnumerateWeights()) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor input)
    {
        int batch = (int)input.Shape[0];
        int height = (int)input.Shape[2];
        int width = (int)input.Shape[3];

        // Left branch: cv1 then sequential bottlenecks.
        Tensor left = _cv1.Forward(backend, input);
        for (int i = 0; i < _numBottlenecks; i++)
        {
            Tensor next = _bottlenecks[i].Forward(backend, left);
            left.Dispose();
            left = next;
        }

        // Right branch: cv2 directly on input.
        Tensor right = _cv2.Forward(backend, input);

        // Concat [left, right] along channel dim → 2 * hiddenChannels.
        TensorShape concatShape = new TensorShape(batch, 2 * _hiddenChannels, height, width);
        Tensor concatenated = new Tensor(concatShape, DType.F32);
        backend.Concat(concatenated, [left, right], dim: 1);
        left.Dispose();
        right.Dispose();

        // cv3 projects to out_channels.
        Tensor output = _cv3.Forward(backend, concatenated);
        concatenated.Dispose();
        return output;
    }
}
