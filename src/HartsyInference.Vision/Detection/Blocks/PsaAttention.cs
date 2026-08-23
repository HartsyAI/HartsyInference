using HartsyInference.Core.Backends;
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
/// <para>The matmul math (custom shapes: q/k have channel <c>kd</c> while v has channel
/// <c>head_dim</c>) doesn't fit <see cref="IBackend.ScaledDotProductAttention"/>'s standard
/// <c>[B,H,S,D]</c> contract, so we hand-code the softmax + matmuls. Sizes are small (N ≤ 1600
/// at 640×640 input) so the inner loop performs adequately on CPU; a future GPU port can fold
/// this into a fused kernel.</para></summary>
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
        _qkvWeight = EnsureF32(weights[$"{prefix}.qkv.conv.weight"]);
        _qkvBias = EnsureF32(weights[$"{prefix}.qkv.conv.bias"]);
        _projWeight = EnsureF32(weights[$"{prefix}.proj.conv.weight"]);
        _projBias = EnsureF32(weights[$"{prefix}.proj.conv.bias"]);
        _peWeight = EnsureF32(weights[$"{prefix}.pe.conv.weight"]);
        _peBias = EnsureF32(weights[$"{prefix}.pe.conv.bias"]);
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
        ComputeAttention(qkv, attnOutCHW, batch, height, width);
        qkv.Dispose();

        Tensor projOut = new Tensor(new TensorShape(batch, _dim, height, width), DType.F32);
        backend.Conv2D(projOut, attnOutCHW, _projWeight!, _projBias!, 1, 1, 0, 0);
        attnOutCHW.Dispose();
        return projOut;
    }

    /// <summary>Computes the attention output (including the depthwise PE on V added inline)
    /// into <paramref name="attnOut"/> shaped <c>[B, dim, H, W]</c>. <paramref name="qkv"/> is
    /// the projected QKV tensor shaped <c>[B, dim + 2*nh*kd, H, W]</c>.</summary>
    private void ComputeAttention(Tensor qkv, Tensor attnOut, int batch, int height, int width)
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

            for (int h = 0; h < _numHeads; h++)
            {
                long headBase = (long)h * slotsPerHead * n;
                float* qPlane = qkvBase + headBase;
                float* kPlane = qPlane + (long)_keyDim * n;
                float* vPlane = kPlane + (long)_keyDim * n;

                // attn[i, j] = scale * sum_d(q[d, i] * k[d, j])
                // Both indexed [d * N + i] (channel-major per head).
                float[] attn = new float[n * n];
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        float sum = 0f;
                        for (int d = 0; d < _keyDim; d++)
                            sum += qPlane[d * n + i] * kPlane[d * n + j];
                        attn[i * n + j] = sum * _scale;
                    }
                    // Softmax over j (numerically stable).
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < n; j++)
                        if (attn[i * n + j] > max) max = attn[i * n + j];
                    float sumExp = 0f;
                    for (int j = 0; j < n; j++)
                    {
                        float e = MathF.Exp(attn[i * n + j] - max);
                        attn[i * n + j] = e;
                        sumExp += e;
                    }
                    float inv = sumExp > 0f ? 1f / sumExp : 0f;
                    for (int j = 0; j < n; j++)
                        attn[i * n + j] *= inv;
                }

                // out[d, i] = sum_j(v[d, j] * attn[i, j])  =  v @ attn^T
                // Place into the output's channel block for this head: out channels h*head_dim .. h*head_dim+head_dim-1.
                long outHeadOffset = (long)h * _headDim * n;
                float* outHead = outBase + outHeadOffset;
                for (int d = 0; d < _headDim; d++)
                {
                    for (int i = 0; i < n; i++)
                    {
                        float sum = 0f;
                        for (int j = 0; j < n; j++)
                            sum += vPlane[d * n + j] * attn[i * n + j];
                        outHead[d * n + i] = sum;
                    }
                }

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

    private static Tensor EnsureF32(Tensor t) =>
        t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}
