using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection.Blocks;

/// <summary>YOLO11 Position-Sensitive multi-head self-attention over flattened spatial tokens.
/// The block has three components:
/// <list type="number">
///   <item><b>QKV projection</b>: single 1×1 conv producing <c>dim + 2 * num_heads * key_dim</c> channels, then split into Q (<c>key_dim</c>/head), K (<c>key_dim</c>/head), V (<c>head_dim</c>/head).</item>
///   <item><b>Attention</b>: <c>softmax(scale * Q^T @ K) @ V^T</c> with the spatial axis acting as the "sequence" axis. Note Q/K have a smaller channel count than V (<c>key_dim = head_dim / 2</c>).</item>
///   <item><b>Positional encoding</b>: 3×3 depthwise conv on the reshaped value tensor, added to the attention output.</item>
///   <item><b>Output projection</b>: 1×1 conv back to the original channel count.</item>
/// </list>
/// <para>All three convs are BN-folded — qkv, proj, and pe each carry weight + bias after the
/// Python converter runs. None of them use SiLU (the parent <see cref="PsaBlock"/> handles
/// activations via its FFN).</para>
/// <para>Q and K have <c>kd</c> channels a head where V has <c>head_dim</c>, which no attention entry takes as it is.
/// The attention pads Q and K with zeros to the value's width and goes through the shared attention layer; the
/// depthwise positional encoding stays a host loop.</para></summary>
public sealed unsafe class PsaAttention
{
    private readonly int _dim;
    private readonly int _numHeads;
    private readonly int _keyDim;
    private readonly int _headDim;
    private readonly float _scale;

    private Tensor? _qkvWeight; // [dim + 2*nh*kd, dim, 1, 1]
    private Tensor? _qkvBias;
    private Tensor? _projWeight; // [dim, dim, 1, 1]
    private Tensor? _projBias;
    private Tensor? _peWeight; // [dim, 1, 3, 3] depthwise
    private Tensor? _peBias;   // [dim]

    public PsaAttention(int dim, int numHeads, float attnRatio = 0.5f)
    {
        if (dim % numHeads != 0)
            throw new ArgumentException($"dim {dim} must be divisible by numHeads {numHeads}.");
        _dim = dim;
        _numHeads = numHeads;
        _headDim = dim / numHeads;
        _keyDim = (int)(_headDim * attnRatio);
        _scale = 1f / MathF.Sqrt(_keyDim);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _qkvWeight = TensorCasts.LoadF32(weights, $"{prefix}.qkv.conv.weight");
        _qkvBias = TensorCasts.LoadF32(weights, $"{prefix}.qkv.conv.bias");
        _projWeight = TensorCasts.LoadF32(weights, $"{prefix}.proj.conv.weight");
        _projBias = TensorCasts.LoadF32(weights, $"{prefix}.proj.conv.bias");
        _peWeight = TensorCasts.LoadF32(weights, $"{prefix}.pe.conv.weight");
        _peBias = TensorCasts.LoadF32(weights, $"{prefix}.pe.conv.bias");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_qkvWeight is not null) yield return _qkvWeight;
        if (_qkvBias is not null) yield return _qkvBias;
        if (_projWeight is not null) yield return _projWeight;
        if (_projBias is not null) yield return _projBias;
        if (_peWeight is not null) yield return _peWeight;
        if (_peBias is not null) yield return _peBias;
    }

    public Tensor Forward(IBackend backend, Tensor input)
    {
        int batch = (int)input.Shape[0];
        int height = (int)input.Shape[2];
        int width = (int)input.Shape[3];
        int n = height * width;
        int qkvChannels = _dim + 2 * _numHeads * _keyDim;

        // 1×1 QKV projection. Output is [B, qkvChannels, H, W].
        Tensor qkv = new Tensor(new TensorShape(batch, qkvChannels, height, width), DType.F32);
        backend.Conv2D(qkv, input, _qkvWeight!, _qkvBias!, 1, 1, 0, 0);

        // Compute attention manually. Allocate intermediates as managed arrays — the spatial
        // dim N is small (≤ 1600 at 640²) so this scratch is cheap.
        Tensor attnOutCHW = new Tensor(new TensorShape(batch, _dim, height, width), DType.F32);
        ComputeAttention(backend, qkv, attnOutCHW, batch, height, width);
        qkv.Dispose();

        Tensor projOut = new Tensor(new TensorShape(batch, _dim, height, width), DType.F32);
        backend.Conv2D(projOut, attnOutCHW, _projWeight!, _projBias!, 1, 1, 0, 0);
        attnOutCHW.Dispose();
        return projOut;
    }

    /// <summary>The attention of one batch item through the shared attention layer. <paramref name="qkv"/> is channel-major
    /// <c>[heads · (2·keyDim + headDim), n]</c>: for each head its query, its key (<paramref name="keyDim"/> channels each)
    /// and its value (<paramref name="headDim"/> channels). The value is wider than the query and key, which no attention
    /// entry takes as it is, so Q and K are zero-padded to the value's width: the added zeros leave every <c>q · k</c>
    /// unchanged, and <paramref name="scale"/> stays the one for <paramref name="keyDim"/>. Returns channel-major
    /// <c>[heads · headDim, n]</c>.</summary>
    internal static float[] Attend(IBackend backend, ReadOnlySpan<float> qkv, int heads, int keyDim, int headDim, int n, float scale)
    {
        int slots = 2 * keyDim + headDim;
        TensorShape shape = new(1, heads, n, headDim);
        Tensor q = new(shape, DType.F32), k = new(shape, DType.F32), v = new(shape, DType.F32);
        Span<float> qs = q.AsSpan<float>(), ks = k.AsSpan<float>(), vs = v.AsSpan<float>();
        qs.Clear();
        ks.Clear();
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < n; i++)
            {
                int row = (h * n + i) * headDim, plane = h * slots * n + i;
                for (int d = 0; d < keyDim; d++)
                {
                    qs[row + d] = qkv[plane + d * n];
                    ks[row + d] = qkv[plane + (keyDim + d) * n];
                }
                for (int d = 0; d < headDim; d++) vs[row + d] = qkv[plane + (2 * keyDim + d) * n];
            }
        // The layer takes q, k and v over and disposes them.
        using Tensor attended = AttentionLayer.Shared(heads, headDim, scale, DType.F32).AttendHeadMajorOperands(backend, q, k, v, default);
        ReadOnlySpan<float> a = attended.AsReadOnlySpan<float>();
        float[] output = new float[heads * headDim * n];
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < n; i++)
                for (int d = 0; d < headDim; d++)
                    output[(h * headDim + d) * n + i] = a[(h * n + i) * headDim + d];
        return output;
    }

    /// <summary>Computes the attention output (including the depthwise PE on V added inline)
    /// into <paramref name="attnOut"/> shaped <c>[B, dim, H, W]</c>. <paramref name="qkv"/> is
    /// the projected QKV tensor shaped <c>[B, dim + 2*nh*kd, H, W]</c>.</summary>
    private void ComputeAttention(IBackend backend, Tensor qkv, Tensor attnOut, int batch, int height, int width)
    {
        int n = height * width;
        int qkvChannels = (int)qkv.Shape[1];
        float* qkvPtr = (float*)qkv.DataPointer;
        float* outPtr = (float*)attnOut.DataPointer;
        float* peWPtr = (float*)_peWeight!.DataPointer;
        float* peBPtr = (float*)_peBias!.DataPointer;

        // The qkv tensor is [B, C', H, W] where channels are interleaved per-head as
        // [head 0 q (kd ch) | head 0 k (kd ch) | head 0 v (head_dim ch) | head 1 q ... ].
        // Ultralytics produces this layout via `.view(B, nh, kd*2+head_dim, N)`.
        // For our flat layout, total per-head channels = kd*2 + head_dim, and channel index for
        // (head h, slot s) is h * (kd*2+head_dim) + s.
        int slotsPerHead = _keyDim * 2 + _headDim;

        for (int b = 0; b < batch; b++)
        {
            long qkvBatchStride = (long)qkvChannels * n;
            long outBatchStride = (long)_dim * n;
            float* qkvBase = qkvPtr + b * qkvBatchStride;
            float* outBase = outPtr + b * outBatchStride;
            float[] attended = Attend(backend, new ReadOnlySpan<float>(qkvBase, qkvChannels * n), _numHeads, _keyDim, _headDim, n, _scale);

            for (int h = 0; h < _numHeads; h++)
            {
                // This head's value planes follow its query and key planes.
                float* vPlane = qkvBase + (long)h * slotsPerHead * n + 2L * _keyDim * n;

                // The attention output for this head: out channels h*head_dim .. h*head_dim+head_dim-1.
                long outHeadOffset = (long)h * _headDim * n;
                float* outHead = outBase + outHeadOffset;
                attended.AsSpan(h * _headDim * n, _headDim * n).CopyTo(new Span<float>(outHead, _headDim * n));

                // Add depthwise PE on the V slice for this head. PE weight shape is [C, 1, 3, 3]
                // — each output channel maps to exactly one input channel (the same index).
                // The V channels live at global output channels [h*head_dim .. h*head_dim+head_dim-1].
                for (int d = 0; d < _headDim; d++)
                {
                    int outChannel = h * _headDim + d;
                    float* kernel = peWPtr + (long)outChannel * 9; // [1, 3, 3] flattened
                    float bias = peBPtr[outChannel];
                    for (int y = 0; y < height; y++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            float v = bias;
                            for (int ky = 0; ky < 3; ky++)
                            {
                                int sy = y + ky - 1;
                                if (sy < 0 || sy >= height) continue;
                                for (int kx = 0; kx < 3; kx++)
                                {
                                    int sx = x + kx - 1;
                                    if (sx < 0 || sx >= width) continue;
                                    v += kernel[ky * 3 + kx] * vPlane[d * n + sy * width + sx];
                                }
                            }
                            outHead[d * n + y * width + x] += v;
                        }
                    }
                }
            }
        }
    }
}
