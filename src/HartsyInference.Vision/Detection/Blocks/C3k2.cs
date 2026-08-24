using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection.Blocks;

/// <summary>YOLO11's evolution of <see cref="C2f"/>. Outer structure is identical (1×1 expand →
/// split-2 → chain inner units → concat-all → 1×1 project), but each inner unit is selected via
/// the <c>c3k</c> flag:
/// <list type="bullet">
///   <item><c>c3k=false</c>: inner unit is a plain <see cref="Bottleneck"/> — same as standard C2f.</item>
///   <item><c>c3k=true</c>: inner unit is a <see cref="C3k"/> (a 2-bottleneck CSP sub-block) — deeper feature mixing for the later backbone stages and the largest neck output.</item>
/// </list>
/// <para>In YOLO11 the early backbone stages and most neck blocks use <c>c3k=false</c> with
/// <c>e=0.25</c> (narrower hidden channels for speed), while the deeper layers (backbone layers 6
/// and 8, and the final neck block at layer 22) use <c>c3k=true</c> with <c>e=0.5</c>.</para></summary>
public sealed class C3k2
{
    private readonly int _outChannels;
    private readonly int _hiddenChannels;
    private readonly int _numInnerUnits;
    private readonly bool _useC3k;

    private readonly ConvBnSilu _cv1;
    private readonly ConvBnSilu _cv2;
    private readonly Bottleneck[]? _bottlenecks; // populated when useC3k = false
    private readonly C3k[]? _c3kBlocks;          // populated when useC3k = true

    public C3k2(int inChannels, int outChannels, int n, bool c3k, bool shortcut, float expansion = 0.5f)
    {
        if (n < 1)
            throw new ArgumentOutOfRangeException(nameof(n), n, "C3k2 requires at least one inner unit.");
        _outChannels = outChannels;
        _hiddenChannels = (int)(outChannels * expansion);
        _numInnerUnits = n;
        _useC3k = c3k;

        _cv1 = new ConvBnSilu(2 * _hiddenChannels, 1, 1, 0, 0, useSilu: true);
        _cv2 = new ConvBnSilu(outChannels, 1, 1, 0, 0, useSilu: true);

        if (c3k)
        {
            _c3kBlocks = new C3k[n];
            for (int i = 0; i < n; i++)
                _c3kBlocks[i] = new C3k(_hiddenChannels, _hiddenChannels, n: 2, shortcut);
        }
        else
        {
            _bottlenecks = new Bottleneck[n];
            // Ultralytics' C3k2 constructs inner Bottlenecks WITHOUT overriding the default e=0.5,
            // so the bottleneck compresses by half (cv1: c → c/2, cv2: c/2 → c). This differs from
            // C2f, which explicitly passes e=1.0. Passing e=0.5 here matches the v11 checkpoints.
            for (int i = 0; i < n; i++)
                _bottlenecks[i] = new Bottleneck(_hiddenChannels, _hiddenChannels, shortcut, expansion: 0.5f);
        }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _cv1.LoadWeights(weights, $"{prefix}.cv1.conv");
        _cv2.LoadWeights(weights, $"{prefix}.cv2.conv");
        if (_useC3k)
        {
            for (int i = 0; i < _c3kBlocks!.Length; i++)
                _c3kBlocks[i].LoadWeights(weights, $"{prefix}.m.{i}");
        }
        else
        {
            for (int i = 0; i < _bottlenecks!.Length; i++)
                _bottlenecks[i].LoadWeights(weights, $"{prefix}.m.{i}");
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _cv1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _cv2.EnumerateWeights()) yield return t;
        if (_useC3k)
        {
            for (int i = 0; i < _c3kBlocks!.Length; i++)
                foreach (Tensor t in _c3kBlocks[i].EnumerateWeights()) yield return t;
        }
        else
        {
            for (int i = 0; i < _bottlenecks!.Length; i++)
                foreach (Tensor t in _bottlenecks[i].EnumerateWeights()) yield return t;
        }
    }

    public Tensor Forward(IBackend backend, Tensor input)
    {
        int batch = (int)input.Shape[0];
        int height = (int)input.Shape[2];
        int width = (int)input.Shape[3];

        Tensor expanded = _cv1.Forward(backend, input);

        Tensor[] accumulated = new Tensor[2 + _numInnerUnits];
        TensorShape halfShape = new TensorShape(batch, _hiddenChannels, height, width);
        accumulated[0] = new Tensor(halfShape, DType.F32);
        accumulated[1] = new Tensor(halfShape, DType.F32);
        backend.Split([accumulated[0], accumulated[1]], expanded, dim: 1);
        expanded.Dispose();

        for (int i = 0; i < _numInnerUnits; i++)
        {
            Tensor prev = accumulated[1 + i];
            accumulated[2 + i] = _useC3k ? _c3kBlocks![i].Forward(backend, prev)
                : _bottlenecks![i].Forward(backend, prev);
        }

        TensorShape concatShape = new TensorShape(batch, (2 + _numInnerUnits) * _hiddenChannels, height, width);
        Tensor concatenated = new Tensor(concatShape, DType.F32);
        backend.Concat(concatenated, accumulated, dim: 1);
        for (int i = 0; i < accumulated.Length; i++)
            accumulated[i].Dispose();

        Tensor output = _cv2.Forward(backend, concatenated);
        concatenated.Dispose();
        return output;
    }
}
