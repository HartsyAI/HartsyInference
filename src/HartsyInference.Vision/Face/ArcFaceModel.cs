using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Face;

/// <summary>InsightFace ArcFace recognition backbone (IResNet, e.g. the IR-50 in buffalo_l's
/// <c>w600k_r50</c>): maps a 112×112 aligned face crop to a 512-d identity embedding. Consumed by
/// IP-Adapter FaceID (which projects the embedding into image-prompt tokens) and usable standalone
/// for face similarity.
///
/// <para>Weights come from <c>tests/python-reference/convert_arcface_onnx.py</c>, which converts the
/// official ONNX release (BN2/BN3/downsample-BN pre-folded into conv biases by the export) and
/// additionally folds the pre-flatten <c>bn2</c> + post-fc <c>features</c> BN1d into <c>fc</c>. Each
/// block's leading <c>bn1</c> survives as a precomputed per-channel <c>scale [C,1,1,1]</c> /
/// <c>shift [C]</c> pair applied via the backend's depthwise 1×1 conv. Block counts per stage are
/// derived from the checkpoint keys, so IR-50 / IR-100 load with the same code.</para>
///
/// <para>IResNet block: <c>bn1 → conv3×3 → PReLU → conv3×3(stride) → add residual</c>, where stage-first
/// blocks carry a 1×1 stride-2 downsample conv on the residual branch. Stem is <c>conv3×3 + PReLU</c>
/// (112→112); stages halve resolution (56/28/14/7) and the head is <c>fc(512·7·7 → 512)</c>.</para></summary>
public sealed class ArcFaceModel
{
    /// <summary>Embedding dimensionality of the released ArcFace checkpoints.</summary>
    public const int EmbeddingDim = 512;

    /// <summary>Expected input crop side (aligned face).</summary>
    public const int InputSize = FaceAlignment.CropSize;

    private sealed class Block
    {
        public required Tensor Bn1Scale;   // [C, 1, 1, 1]
        public required Tensor Bn1Shift;   // [C]
        public required Tensor Conv1Weight;
        public required Tensor Conv1Bias;
        public required Tensor PreluAlpha;
        public required Tensor Conv2Weight;
        public required Tensor Conv2Bias;
        public Tensor? DownsampleWeight;
        public Tensor? DownsampleBias;
        public int Stride => DownsampleWeight is not null ? 2 : 1;
    }

    private Tensor? _conv1Weight;
    private Tensor? _conv1Bias;
    private Tensor? _prelu1Alpha;
    private Tensor? _fcWeight;
    private Tensor? _fcBias;
    private readonly List<List<Block>> _stages = [];

    /// <summary>Loads the converted ArcFace safetensors dict, deriving the per-stage block counts from the keys.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _conv1Weight = F32(weights, "conv1.weight");
        _conv1Bias = F32(weights, "conv1.bias");
        _prelu1Alpha = F32(weights, "prelu1.alpha");
        _fcWeight = F32(weights, "fc.weight");
        _fcBias = F32(weights, "fc.bias");

        _stages.Clear();
        for (int stage = 1; weights.ContainsKey($"layer{stage}.0.conv1.weight"); stage++)
        {
            List<Block> blocks = [];
            for (int b = 0; weights.ContainsKey($"layer{stage}.{b}.conv1.weight"); b++)
            {
                string p = $"layer{stage}.{b}";
                Block block = new()
                {
                    Bn1Scale = F32(weights, $"{p}.bn1.scale"),
                    Bn1Shift = F32(weights, $"{p}.bn1.shift"),
                    Conv1Weight = F32(weights, $"{p}.conv1.weight"),
                    Conv1Bias = F32(weights, $"{p}.conv1.bias"),
                    PreluAlpha = F32(weights, $"{p}.prelu.alpha"),
                    Conv2Weight = F32(weights, $"{p}.conv2.weight"),
                    Conv2Bias = F32(weights, $"{p}.conv2.bias"),
                };
                if (weights.TryGetValue($"{p}.downsample.weight", out Tensor? dsW))
                {
                    block.DownsampleWeight = dsW.DType == DType.F32 ? dsW : dsW.CastTo(DType.F32);
                    block.DownsampleBias = F32(weights, $"{p}.downsample.bias");
                }
                blocks.Add(block);
            }
            _stages.Add(blocks);
        }
        if (_stages.Count == 0)
            throw new InvalidOperationException("ArcFace checkpoint has no layer{N}.0.conv1.weight keys — wrong file or unconverted checkpoint (run convert_arcface_onnx.py).");
    }

    /// <summary>Converts an aligned 112×112 HWC RGB u8 crop to the model input <c>[1,3,112,112]</c> with
    /// insightface's ArcFace normalization <c>(v − 127.5) / 127.5</c>.</summary>
    public static unsafe Tensor PreprocessAligned(ReadOnlySpan<byte> rgb112)
    {
        const int size = InputSize;
        if (rgb112.Length != size * size * 3)
            throw new ArgumentException($"Expected a {size}×{size} RGB crop ({size * size * 3} bytes); got {rgb112.Length}.");
        Tensor input = new(new TensorShape(1, 3, size, size), DType.F32);
        float* p = (float*)input.DataPointer;
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                int pix = y * size + x;
                for (int c = 0; c < 3; c++)
                    p[c * size * size + pix] = (rgb112[pix * 3 + c] - 127.5f) / 127.5f;
            }
        }
        return input;
    }

    /// <summary>Runs the backbone on <c>[B,3,112,112]</c> normalized input and returns the raw (un-normalized)
    /// <c>[B,512]</c> embedding. Caller owns the returned tensor. <paramref name="stageProbe"/> receives
    /// intermediate activations (stem / layer{N}) for parity debugging; production callers pass null.</summary>
    public Tensor Forward(IBackend backend, Tensor input, Action<string, Tensor>? stageProbe = null)
    {
        if (_conv1Weight is null || _fcWeight is null)
            throw new InvalidOperationException("ArcFaceModel weights not loaded.");
        if (input.Shape.Rank != 4 || input.Shape[1] != 3)
            throw new ArgumentException($"ArcFace input must be [B,3,{InputSize},{InputSize}]; got {input.Shape}.");

        int batch = (int)input.Shape[0];
        int h = (int)input.Shape[2];
        int w = (int)input.Shape[3];

        Tensor stem = new(new TensorShape(batch, 64, h, w), DType.F32);
        backend.Conv2D(stem, input, _conv1Weight, _conv1Bias, 1, 1, 1, 1);
        Tensor x = ApplyPrelu(backend, stem, _prelu1Alpha!);
        stem.Dispose();
        stageProbe?.Invoke("stem", x);

        for (int s = 0; s < _stages.Count; s++)
        {
            foreach (Block block in _stages[s])
            {
                Tensor next = ForwardBlock(backend, x, block);
                x.Dispose();
                x = next;
            }
            stageProbe?.Invoke($"layer{s + 1}", x);
        }

        long flat = x.Shape[1] * x.Shape[2] * x.Shape[3];
        if (flat != _fcWeight.Shape[1])
            throw new InvalidOperationException($"ArcFace feature map {x.Shape} flattens to {flat}, but fc expects {_fcWeight.Shape[1]} — input size must be {InputSize}×{InputSize}.");
        Tensor flatView = x.Reshape(new TensorShape(batch, flat));
        Tensor embedding = new(new TensorShape(batch, EmbeddingDim), DType.F32);
        backend.Linear(embedding, flatView, _fcWeight, _fcBias);
        x.Dispose();
        return embedding;
    }

    /// <summary>Runs <see cref="Forward"/> and L2-normalizes each row on the host — the "normed embedding"
    /// IP-Adapter FaceID and face-similarity comparisons consume.</summary>
    public unsafe Tensor EmbedNormalized(IBackend backend, Tensor input)
    {
        Tensor embedding = Forward(backend, input);
        int batch = (int)embedding.Shape[0];
        float* p = (float*)embedding.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            float sumSq = 0f;
            for (int d = 0; d < EmbeddingDim; d++) sumSq += p[b * EmbeddingDim + d] * p[b * EmbeddingDim + d];
            float inv = 1.0f / MathF.Sqrt(MathF.Max(sumSq, 1e-12f));
            for (int d = 0; d < EmbeddingDim; d++) p[b * EmbeddingDim + d] *= inv;
        }
        return embedding;
    }

    /// <summary>Yields every weight tensor for GPU preload / free.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_conv1Weight is not null) yield return _conv1Weight;
        if (_conv1Bias is not null) yield return _conv1Bias;
        if (_prelu1Alpha is not null) yield return _prelu1Alpha;
        foreach (List<Block> stage in _stages)
        {
            foreach (Block b in stage)
            {
                yield return b.Bn1Scale;
                yield return b.Bn1Shift;
                yield return b.Conv1Weight;
                yield return b.Conv1Bias;
                yield return b.PreluAlpha;
                yield return b.Conv2Weight;
                yield return b.Conv2Bias;
                if (b.DownsampleWeight is not null) yield return b.DownsampleWeight;
                if (b.DownsampleBias is not null) yield return b.DownsampleBias;
            }
        }
        if (_fcWeight is not null) yield return _fcWeight;
        if (_fcBias is not null) yield return _fcBias;
    }

    private static Tensor ForwardBlock(IBackend backend, Tensor input, Block block)
    {
        int batch = (int)input.Shape[0];
        int inH = (int)input.Shape[2];
        int inW = (int)input.Shape[3];
        int channels = (int)block.Conv1Weight.Shape[0];
        int stride = block.Stride;
        int outH = (inH + 2 - 3) / stride + 1;
        int outW = (inW + 2 - 3) / stride + 1;

        // bn1 as a per-channel affine — a 1×1 depthwise conv with weight [C,1,1,1] + bias [C].
        Tensor bn = new(input.Shape, DType.F32);
        backend.Conv2dDepthwise(bn, input, block.Bn1Scale, block.Bn1Shift, 1, 1, 0, 0);

        Tensor mid = new(new TensorShape(batch, channels, inH, inW), DType.F32);
        backend.Conv2D(mid, bn, block.Conv1Weight, block.Conv1Bias, 1, 1, 1, 1);
        bn.Dispose();
        Tensor act = ApplyPrelu(backend, mid, block.PreluAlpha);
        mid.Dispose();

        Tensor main = new(new TensorShape(batch, channels, outH, outW), DType.F32);
        backend.Conv2D(main, act, block.Conv2Weight, block.Conv2Bias, stride, stride, 1, 1);
        act.Dispose();

        Tensor sum = new(new TensorShape(batch, channels, outH, outW), DType.F32);
        if (block.DownsampleWeight is not null)
        {
            Tensor shortcut = new(new TensorShape(batch, channels, outH, outW), DType.F32);
            backend.Conv2D(shortcut, input, block.DownsampleWeight, block.DownsampleBias, stride, stride, 0, 0);
            backend.Add(sum, main, shortcut);
            shortcut.Dispose();
        }
        else
        {
            backend.Add(sum, main, input);
        }
        main.Dispose();
        return sum;
    }

    /// <summary>Per-channel PReLU on an NCHW tensor via the backend's [B,C,T] op (T = H·W). Writes into a fresh
    /// output (never in-place through a reshape view — a view is a distinct Tensor object, so an in-place op
    /// through it would leave the parent's GPU activation-cache entry stale).</summary>
    private static Tensor ApplyPrelu(IBackend backend, Tensor x, Tensor alpha)
    {
        TensorShape shape3 = new(x.Shape[0], x.Shape[1], x.Shape[2] * x.Shape[3]);
        Tensor input3 = x.Reshape(shape3);
        Tensor output3 = new(shape3, DType.F32);
        backend.Prelu(output3, input3, alpha);
        Tensor output4 = output3.Reshape(x.Shape);
        // The rank-4 view roots the rank-3 owner (SetKeepAlive), so returning the view is safe.
        return output4;
    }

    private static Tensor F32(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        if (!weights.TryGetValue(key, out Tensor? t))
            throw new KeyNotFoundException($"ArcFace checkpoint missing '{key}' — run convert_arcface_onnx.py to produce the folded layout.");
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }
}
