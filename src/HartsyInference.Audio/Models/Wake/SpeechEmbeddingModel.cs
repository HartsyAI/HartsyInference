using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Wake;

/// <summary>Google's <c>speech_embedding</c> backbone (arXiv:2002.01322), ported from openWakeWord's
/// <c>embedding_model.onnx</c>. Turns a 76×32 mel window (~775 ms) into one 96-d embedding. Frozen and shared
/// by every wake word, so adding a phrase costs only its head — this is why many wake words run for barely
/// more than one.
///
/// <para>Five stages of <c>1×3</c> (temporal) and <c>3×1</c> (frequency) convolutions with channels
/// 24 → 48 → 72 → 96, pooled between stages. BatchNorm is already folded into the conv weights upstream
/// (<c>*_fused_bn</c>), so there is no norm to apply here.</para>
///
/// <para>The activation is a <b>clipped</b> LeakyReLU — <c>max(leaky_relu(x, 0.2), −0.4)</c> — not the plain
/// ReLU the model is usually described with. Using ReLU loads and runs fine and simply scores worse, so it
/// is worth stating: this is the kind of mistake that shows up as accuracy, never as an error.</para>
///
/// <para>Stateless: all streaming state lives in the caller's mel and feature ring buffers.</para></summary>
public sealed unsafe class SpeechEmbeddingModel : IDisposable
{
    /// <summary>Mel frames in one embedding window.</summary>
    public const int WindowFrames = 76;
    /// <summary>Mel frames advanced between consecutive embeddings (80 ms).</summary>
    public const int StrideFrames = 8;
    /// <summary>Embedding width.</summary>
    public const int EmbeddingDim = 96;

    private const float LeakySlope = 0.2f;
    private const float ActivationFloor = -0.4f;

    private readonly List<Tensor> _owned = [];
    private ConvW[] _convs = [];
    private ConvW _final;
    private int _disposed;

    /// <summary>Kernel, padding and pooling for each stage, in graph order. Height is time, width is frequency;
    /// <c>3×1</c> convolutions are unpadded so time shrinks by 2, and <c>1×3</c> are padded so frequency is kept.</summary>
    private static readonly (int Kh, int Kw, int PadH, int PadW)[] ConvSpecs =
    [
        (3, 3, 0, 1), (1, 3, 0, 1), (3, 1, 0, 0),
        (1, 3, 0, 1), (3, 1, 0, 0), (1, 3, 0, 1), (3, 1, 0, 0),
        (1, 3, 0, 1), (3, 1, 0, 0), (1, 3, 0, 1), (3, 1, 0, 0),
        (1, 3, 0, 1), (3, 1, 0, 0), (1, 3, 0, 1), (3, 1, 0, 0),
        (1, 3, 0, 1), (3, 1, 0, 0), (1, 3, 0, 1), (3, 1, 0, 0),
    ];

    /// <summary>Pooling applied AFTER the conv at each index, as <c>(kernelH, kernelW, strideH, strideW)</c>.</summary>
    private static readonly Dictionary<int, (int Kh, int Kw, int Sh, int Sw)> PoolAfter = new()
    {
        [2] = (2, 2, 2, 2),
        [6] = (1, 2, 1, 2),
        [10] = (2, 2, 2, 2),
        [14] = (1, 2, 1, 2),
        [18] = (2, 2, 2, 2),
    };

    /// <summary>Loads the 41 conv tensors from <c>embedding_model.onnx</c>'s initializers.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _convs = new ConvW[ConvSpecs.Length];
        for (int i = 0; i < ConvSpecs.Length; i++)
        {
            string name = i == 0 ? "model/conv2d/Conv2D" : $"model/conv2d_{i}/Conv2D";
            _convs[i] = new ConvW(Track(Get(weights, name + "_weights_fused_bn")), Track(Get(weights, name + "_bias_fused_bn")));
        }
        _final = new ConvW(Track(Get(weights, "model/conv2d_19/Conv2D/ReadVariableOp:0")), null);
    }

    private Tensor Track(Tensor t)
    {
        _owned.Add(t);
        return t;
    }

    private static Tensor Get(IReadOnlyDictionary<string, Tensor> weights, string name) =>
        weights.TryGetValue(name, out Tensor? t)
            ? WhisperOps.EnsureF32(t)
            : throw new HartsyInferenceException($"Speech embedding backbone is missing '{name}'. Expected openWakeWord's embedding_model.onnx.");

    /// <summary>Embeds one mel window. <paramref name="melWindow"/> is <c>76×32</c> row-major (frame-major);
    /// <paramref name="output"/> receives 96 floats.</summary>
    public void Forward(IBackend backend, ReadOnlySpan<float> melWindow, Span<float> output)
    {
        if (_convs.Length == 0) throw new InvalidOperationException("SpeechEmbeddingModel weights not loaded.");
        if (melWindow.Length != WindowFrames * WakeMelFrontend.MelBins)
            throw new ArgumentException($"Embedding window must be {WindowFrames * WakeMelFrontend.MelBins} floats, got {melWindow.Length}.", nameof(melWindow));
        if (output.Length < EmbeddingDim)
            throw new ArgumentException($"Embedding output needs {EmbeddingDim} floats, got {output.Length}.", nameof(output));

        Tensor x = new(new TensorShape(1, 1, WindowFrames, WakeMelFrontend.MelBins), DType.F32);
        melWindow.CopyTo(x.AsSpan<float>());

        for (int i = 0; i < _convs.Length; i++)
        {
            (int kh, int kw, int padH, int padW) = ConvSpecs[i];
            x = Conv(backend, x, _convs[i], kh, kw, padH, padW);
            Activate(x);
            if (PoolAfter.TryGetValue(i, out (int Kh, int Kw, int Sh, int Sw) pool))
                x = Pool(backend, x, pool.Kh, pool.Kw, pool.Sh, pool.Sw);
        }

        // The last convolution collapses the remaining 3 time steps and has neither bias nor activation.
        x = Conv(backend, x, _final, 3, 1, 0, 0);
        if (x.Shape[1] != EmbeddingDim || x.Shape[2] != 1 || x.Shape[3] != 1)
            throw new HartsyInferenceException($"Speech embedding backbone produced {x.Shape}, expected [1,{EmbeddingDim},1,1].");

        x.AsSpan<float>()[..EmbeddingDim].CopyTo(output);
        x.Dispose();
    }

    private static Tensor Conv(IBackend backend, Tensor x, ConvW w, int kh, int kw, int padH, int padW)
    {
        int outC = (int)w.Weight.Shape[0];
        int h = (int)x.Shape[2], width = (int)x.Shape[3];
        int outH = h + 2 * padH - kh + 1, outW = width + 2 * padW - kw + 1;
        Tensor o = new(new TensorShape(1, outC, outH, outW), DType.F32);
        backend.Conv2D(o, x, w.Weight, w.Bias, 1, 1, padH, padW);
        x.Dispose();
        return o;
    }

    private static Tensor Pool(IBackend backend, Tensor x, int kh, int kw, int sh, int sw)
    {
        int c = (int)x.Shape[1], h = (int)x.Shape[2], width = (int)x.Shape[3];
        Tensor o = new(new TensorShape(1, c, (h - kh) / sh + 1, (width - kw) / sw + 1), DType.F32);
        backend.MaxPool2D(o, x, kh, kw, sh, sw, 0, 0);
        x.Dispose();
        return o;
    }

    private static void Activate(Tensor x)
    {
        float* p = (float*)x.DataPointer;
        long n = x.Shape.ElementCount;
        for (long i = 0; i < n; i++)
        {
            float v = p[i];
            if (v < 0f) v *= LeakySlope;
            p[i] = v < ActivationFloor ? ActivationFloor : v;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
    }

    private readonly record struct ConvW(Tensor Weight, Tensor? Bias);
}
