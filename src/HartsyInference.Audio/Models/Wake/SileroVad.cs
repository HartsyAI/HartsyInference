using HartsyInference.Audio.Layers;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Wake;

/// <summary>Silero VAD (MIT), ported from the <c>silero_vad.onnx</c> 16 kHz branch. Scores one 512-sample
/// (32 ms) chunk at a time and returns the probability that it contains speech.
///
/// <para>A fixed DFT basis stored as a 258-filter Conv1d produces 129 real and 129 imaginary bins; the
/// front-end takes their <b>magnitude</b> (<c>sqrt(re²+im²)</c>), not the power spectrum. Four ReLU
/// convolutions with strides 1/2/2/1 then collapse the 4 spectrogram frames to exactly one, which is why the
/// recurrence is a single LSTM cell step rather than a sequence pass.</para>
///
/// <para>The graph applies a <b>ReLU to the LSTM hidden state</b> before the final 1×1 convolution. It is
/// easy to miss — the model still runs without it and simply scores differently.</para>
///
/// <para>Audio must be normalized to ±1, unlike the wake-word front-end which takes int16 scale. The 64
/// samples of context prepended to each chunk are carried internally, so callers push bare 512-sample chunks
/// and call <see cref="Reset"/> on a stream discontinuity. One instance holds one stream's state and is not
/// thread-safe; give each concurrent session its own.</para></summary>
public sealed unsafe class SileroVad : IDisposable
{
    /// <summary>Samples scored per step (32 ms at 16 kHz).</summary>
    public const int WindowSamples = 512;
    /// <summary>Samples of the previous chunk prepended to each window.</summary>
    public const int ContextSamples = 64;
    /// <summary>Total samples the network consumes per step.</summary>
    public const int InputSamples = ContextSamples + WindowSamples;
    /// <summary>LSTM hidden width, and the encoder's output channel count.</summary>
    public const int HiddenDim = 128;

    private const int FftSize = 256;
    private const int HopSamples = 128;
    private const int SpecBins = 129;
    // nn.ReflectionPad1d((0, 64)) — the window is mirrored on the right only, which lands the STFT on 4 frames.
    private const int PadRight = 64;
    private const int PaddedSamples = InputSamples + PadRight;
    private const int StftFrames = (PaddedSamples - FftSize) / HopSamples + 1;

    private readonly List<Tensor> _owned = [];
    private readonly float[] _context = new float[ContextSamples];

    private readonly Tensor _padded = new(new TensorShape(1, 1, PaddedSamples), DType.F32);
    private readonly Tensor _spectrum = new(new TensorShape(1, 2 * SpecBins, StftFrames), DType.F32);
    private readonly Tensor _magnitude = new(new TensorShape(1, SpecBins, StftFrames), DType.F32);
    private readonly Tensor _enc1 = new(new TensorShape(1, 128, StftFrames), DType.F32);
    private readonly Tensor _enc1Act = new(new TensorShape(1, 128, StftFrames), DType.F32);
    private readonly Tensor _enc2 = new(new TensorShape(1, 64, 2), DType.F32);
    private readonly Tensor _enc2Act = new(new TensorShape(1, 64, 2), DType.F32);
    private readonly Tensor _enc3 = new(new TensorShape(1, 64, 1), DType.F32);
    private readonly Tensor _enc3Act = new(new TensorShape(1, 64, 1), DType.F32);
    private readonly Tensor _enc4 = new(new TensorShape(1, HiddenDim, 1), DType.F32);
    private readonly Tensor _enc4Act = new(new TensorShape(1, HiddenDim, 1), DType.F32);
    private readonly Tensor _lstmInput = new(new TensorShape(1, 1, HiddenDim), DType.F32);
    private readonly Tensor _gatesInput = new(new TensorShape(1, 1, 4 * HiddenDim), DType.F32);
    private readonly Tensor _gatesHidden = new(new TensorShape(1, 1, 4 * HiddenDim), DType.F32);
    private readonly Tensor _gates = new(new TensorShape(1, 1, 4 * HiddenDim), DType.F32);
    private readonly Tensor _headInput = new(new TensorShape(1, HiddenDim, 1), DType.F32);
    private readonly Tensor _logit = new(new TensorShape(1, 1, 1), DType.F32);
    private readonly Tensor _probability = new(new TensorShape(1, 1, 1), DType.F32);

    private Tensor _hidden = new(new TensorShape(1, 1, HiddenDim), DType.F32);
    private Tensor _cell = new(new TensorShape(1, 1, HiddenDim), DType.F32);
    private Tensor _hiddenNext = new(new TensorShape(1, 1, HiddenDim), DType.F32);
    private Tensor _cellNext = new(new TensorShape(1, 1, HiddenDim), DType.F32);

    private Tensor? _stft;
    private ConvW _conv1, _conv2, _conv3, _conv4, _final;
    private Tensor? _weightIh, _weightHh, _biasIh, _biasHh;
    private int _disposed;

    /// <summary>Binds the 15 tensors of <c>silero_vad_16k.safetensors</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _stft = Track(Get(weights, "stft_conv.weight"));
        _conv1 = LoadConv(weights, "conv1");
        _conv2 = LoadConv(weights, "conv2");
        _conv3 = LoadConv(weights, "conv3");
        _conv4 = LoadConv(weights, "conv4");
        _final = LoadConv(weights, "final_conv");
        _weightIh = Track(Get(weights, "lstm_cell.weight_ih"));
        _weightHh = Track(Get(weights, "lstm_cell.weight_hh"));
        _biasIh = Track(Get(weights, "lstm_cell.bias_ih"));
        _biasHh = Track(Get(weights, "lstm_cell.bias_hh"));
        Reset();
    }

    /// <summary>Zeros the LSTM state and the carried audio context. Call whenever the stream jumps.</summary>
    public void Reset()
    {
        _hidden.AsSpan<float>().Clear();
        _cell.AsSpan<float>().Clear();
        Array.Clear(_context);
    }

    /// <summary>Speech probability for one 512-sample chunk of audio normalized to ±1.</summary>
    public float Process(IBackend backend, ReadOnlySpan<float> chunk)
    {
        if (_stft is null || _weightIh is null || _weightHh is null || _biasIh is null || _biasHh is null)
            throw new InvalidOperationException("SileroVad weights not loaded.");
        if (chunk.Length != WindowSamples)
            throw new ArgumentException($"SileroVad takes {WindowSamples} samples per step, got {chunk.Length}.", nameof(chunk));

        Span<float> padded = _padded.AsSpan<float>();
        _context.CopyTo(padded);
        chunk.CopyTo(padded[ContextSamples..]);
        for (int i = 0; i < PadRight; i++) padded[InputSamples + i] = padded[InputSamples - 2 - i];
        chunk[^ContextSamples..].CopyTo(_context);

        backend.Conv1d(_spectrum, _padded, _stft, null, HopSamples, 0, 0, 1, 1);
        Magnitude();

        Encode(backend, _enc1, _enc1Act, _magnitude, _conv1, 1);
        Encode(backend, _enc2, _enc2Act, _enc1Act, _conv2, 2);
        Encode(backend, _enc3, _enc3Act, _enc2Act, _conv3, 2);
        Encode(backend, _enc4, _enc4Act, _enc3Act, _conv4, 1);

        // LstmOps carries PyTorch's i,f,g,o gate order, which is how the safetensors stores the two matrices.
        _enc4Act.AsSpan<float>().CopyTo(_lstmInput.AsSpan<float>());
        backend.Linear(_gatesInput, _lstmInput, _weightIh, _biasIh);
        backend.Linear(_gatesHidden, _hidden, _weightHh, _biasHh);
        backend.Add(_gates, _gatesInput, _gatesHidden);
        LstmOps.GateAndUpdate(_gates, _cell, _hiddenNext, _cellNext, 1, HiddenDim);
        (_hidden, _hiddenNext) = (_hiddenNext, _hidden);
        (_cell, _cellNext) = (_cellNext, _cell);

        backend.LeakyRelu(_headInput, _hidden, 0f);
        backend.Conv1d(_logit, _headInput, _final.Weight, _final.Bias, 1, 0, 0, 1, 1);
        backend.Sigmoid(_probability, _logit);
        return _probability.AsSpan<float>()[0];
    }

    /// <summary>Collapses the 129 real and 129 imaginary DFT channels to their magnitude. There is no
    /// <c>IBackend</c> complex-magnitude op and the sweep is 516 floats, so it runs on the CPU pointers.</summary>
    private void Magnitude()
    {
        float* spec = (float*)_spectrum.DataPointer;
        float* mag = (float*)_magnitude.DataPointer;
        for (int bin = 0; bin < SpecBins; bin++)
        {
            int real = bin * StftFrames, imaginary = (SpecBins + bin) * StftFrames;
            for (int t = 0; t < StftFrames; t++)
                mag[real + t] = MathF.Sqrt(spec[real + t] * spec[real + t] + spec[imaginary + t] * spec[imaginary + t]);
        }
    }

    private static void Encode(IBackend backend, Tensor conv, Tensor activated, Tensor input, ConvW w, int stride)
    {
        backend.Conv1d(conv, input, w.Weight, w.Bias, stride, 1, 1, 1, 1);
        backend.LeakyRelu(activated, conv, 0f);
    }

    private ConvW LoadConv(IReadOnlyDictionary<string, Tensor> weights, string prefix) =>
        new(Track(Get(weights, prefix + ".weight")), Track(Get(weights, prefix + ".bias")));

    private Tensor Track(Tensor t)
    {
        _owned.Add(t);
        return t;
    }

    private static Tensor Get(IReadOnlyDictionary<string, Tensor> weights, string name) =>
        WakeWeights.Require(weights, name, "silero-vad's silero_vad_16k.safetensors");

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
        Tensor[] scratch =
        [
            _padded, _spectrum, _magnitude, _enc1, _enc1Act, _enc2, _enc2Act, _enc3, _enc3Act, _enc4, _enc4Act,
            _lstmInput, _gatesInput, _gatesHidden, _gates, _headInput, _logit, _probability,
            _hidden, _cell, _hiddenNext, _cellNext,
        ];
        foreach (Tensor t in scratch) t.Dispose();
    }

    private readonly record struct ConvW(Tensor Weight, Tensor Bias);
}
