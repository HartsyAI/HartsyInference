using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Vision.Detection.Blocks;

/// <summary>YOLO11's Cross-Stage Partial with Position-Sensitive Attention. The structure mirrors
/// <see cref="C2f"/> on the outside but the inner chain is a sequence of <see cref="PsaBlock"/>
/// instances applied to only one half of the channel split:
/// <list type="number">
///   <item>1×1 <c>cv1</c> expands input from <c>c</c> to <c>2c</c> channels.</item>
///   <item>Split along channels into <c>a</c> (passthrough) and <c>b</c> (transformed).</item>
///   <item>Run <c>n</c> sequential <see cref="PsaBlock"/> instances on <c>b</c>.</item>
///   <item>Concatenate <c>[a, b]</c> and project through 1×1 <c>cv2</c>.</item>
/// </list>
/// <para>Channel math: <c>self.c = int(c1 * 0.5)</c> sets the hidden width. <c>num_heads</c> is
/// derived as <c>self.c // 64</c> per Ultralytics' default. For YOLO11n the SPPF output is 256
/// channels, so <c>self.c = 128</c> and <c>num_heads = 2</c>.</para></summary>
public sealed class C2psa
{
    private readonly int _inOutChannels;
    private readonly int _hiddenChannels;
    private readonly int _numHeads;
    private readonly int _numBlocks;

    private readonly ConvBnSilu _cv1;
    private readonly ConvBnSilu _cv2;
    private readonly PsaBlock[] _blocks;

    /// <summary>Creates a C2PSA. <paramref name="inOutChannels"/> must equal the input channel
    /// count (Ultralytics asserts c1 == c2). <paramref name="n"/> controls how many PSA blocks
    /// run on the transformed half.</summary>
    public C2psa(int inOutChannels, int n, float expansion = 0.5f)
    {
        if (n < 1)
            throw new ArgumentOutOfRangeException(nameof(n), n, "C2PSA requires at least one PSA block.");
        _inOutChannels = inOutChannels;
        _hiddenChannels = (int)(inOutChannels * expansion);
        _numHeads = Math.Max(1, _hiddenChannels / 64);
        _numBlocks = n;

        _cv1 = new ConvBnSilu(2 * _hiddenChannels, 1, 1, 0, 0, useSilu: true);
        _cv2 = new ConvBnSilu(inOutChannels, 1, 1, 0, 0, useSilu: true);
        _blocks = new PsaBlock[n];
        for (int i = 0; i < n; i++)
            _blocks[i] = new PsaBlock(_hiddenChannels, _numHeads, attnRatio: 0.5f, shortcut: true);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _cv1.LoadWeights(weights, $"{prefix}.cv1.conv");
        _cv2.LoadWeights(weights, $"{prefix}.cv2.conv");
        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i].LoadWeights(weights, $"{prefix}.m.{i}");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _cv1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _cv2.EnumerateWeights()) yield return t;
        for (int i = 0; i < _blocks.Length; i++)
            foreach (Tensor t in _blocks[i].EnumerateWeights()) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor input)
    {
        int batch = (int)input.Shape[0];
        int height = (int)input.Shape[2];
        int width = (int)input.Shape[3];

        // Expand to 2c and split.
        Tensor expanded = _cv1.Forward(backend, input);
        TensorShape halfShape = new TensorShape(batch, _hiddenChannels, height, width);
        Tensor partA = new Tensor(halfShape, DType.F32);
        Tensor partB = new Tensor(halfShape, DType.F32);
        backend.Split([partA, partB], expanded, dim: 1);
        expanded.Dispose();

        // Sequential PSA on partB.
        Tensor current = partB;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, current);
            current.Dispose();
            current = next;
        }

        // Concat [a, b'] → 2c, project to inOutChannels.
        TensorShape concatShape = new TensorShape(batch, 2 * _hiddenChannels, height, width);
        Tensor concatenated = new Tensor(concatShape, DType.F32);
        backend.Concat(concatenated, [partA, current], dim: 1);
        partA.Dispose();
        current.Dispose();

        Tensor output = _cv2.Forward(backend, concatenated);
        concatenated.Dispose();
        return output;
    }
}
