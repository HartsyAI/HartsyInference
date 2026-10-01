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
/// <para>Every convolution here is at most four steps long, so each runs as a matrix product rather than
/// through the generic Conv1d kernel, whose per-tap bookkeeping dominates at that length: the STFT over the four
/// hop-spaced windows, and each encoder layer over its unfolded (im2col) input. Activations are kept time-major
/// (<c>[T, C]</c>) for that; the weights are only viewed, as <c>[C_out, C_in·K]</c>, never copied.</para>
///
/// <para>Audio must be normalized to ±1, unlike the wake-word front-end which takes int16 scale. The 64
/// samples of context prepended to each chunk are carried internally, so callers push bare 512-sample chunks
/// and call <see cref="Reset"/> on a stream discontinuity. One instance holds one stream's state and is not
/// thread-safe; give each concurrent session its own.</para></summary>
public sealed unsafe class SileroVad : IVadModel, IDisposable
{
    /// <summary>Samples scored per step (32 ms at 16 kHz).</summary>
    public const int WindowSamples = 512;

    /// <inheritdoc/>
    int IVadModel.WindowSamples => WindowSamples;
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
    // Every encoder conv is kernel 3 with padding 1.
    private const int Kernel = 3;

    private readonly List<Tensor> _owned = [];
    private readonly float[] _context = new float[ContextSamples];
    private readonly float[] _padded = new float[PaddedSamples];

    private readonly Tensor _frames = new(new TensorShape(StftFrames, FftSize), DType.F32);
    private readonly Tensor _spectrum = new(new TensorShape(StftFrames, 2 * SpecBins), DType.F32);
    private readonly Tensor _magnitude = new(new TensorShape(StftFrames, SpecBins), DType.F32);
    private readonly Encoder _enc1 = new(StftFrames, SpecBins, 128, 1);
    private readonly Encoder _enc2 = new(StftFrames, 128, 64, 2);
    private readonly Encoder _enc3 = new(2, 64, 64, 2);
    private readonly Encoder _enc4 = new(1, 64, HiddenDim, 1);
    private readonly Tensor _gatesInput = new(new TensorShape(1, 4 * HiddenDim), DType.F32);
    private readonly Tensor _gatesHidden = new(new TensorShape(1, 4 * HiddenDim), DType.F32);
    private readonly Tensor _gates = new(new TensorShape(1, 4 * HiddenDim), DType.F32);
    private readonly Tensor _headInput = new(new TensorShape(1, HiddenDim, 1), DType.F32);
    private readonly Tensor _logit = new(new TensorShape(1, 1, 1), DType.F32);
    private readonly Tensor _probability = new(new TensorShape(1, 1, 1), DType.F32);

    private Tensor _hidden = new(new TensorShape(1, HiddenDim), DType.F32);
    private Tensor _cell = new(new TensorShape(1, HiddenDim), DType.F32);
    private Tensor _hiddenNext = new(new TensorShape(1, HiddenDim), DType.F32);
    private Tensor _cellNext = new(new TensorShape(1, HiddenDim), DType.F32);

    private Tensor? _stftMatrix;
    private ConvW _final;
    private Tensor? _weightIh, _weightHh, _biasIh, _biasHh;
    private int _disposed;

    /// <summary>Binds the 15 tensors of <c>silero_vad_16k.safetensors</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        Tensor stft = Track(Get(weights, "stft_conv.weight"));
        _stftMatrix = stft.Reshape(new TensorShape(2 * SpecBins, FftSize));
        _enc1.Bind(LoadConv(weights, "conv1"));
        _enc2.Bind(LoadConv(weights, "conv2"));
        _enc3.Bind(LoadConv(weights, "conv3"));
        _enc4.Bind(LoadConv(weights, "conv4"));
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
        if (_stftMatrix is null || _weightIh is null || _weightHh is null || _biasIh is null || _biasHh is null)
            throw new InvalidOperationException("SileroVad weights not loaded.");
        if (chunk.Length != WindowSamples)
            throw new ArgumentException($"SileroVad takes {WindowSamples} samples per step, got {chunk.Length}.", nameof(chunk));

        Span<float> padded = _padded;
        _context.CopyTo(padded);
        chunk.CopyTo(padded[ContextSamples..]);
        for (int i = 0; i < PadRight; i++) padded[InputSamples + i] = padded[InputSamples - 2 - i];
        chunk[^ContextSamples..].CopyTo(_context);

        // The stride-128 STFT conv as one product: the four hop-spaced windows against the 258 DFT filters.
        Span<float> frames = _frames.AsSpan<float>();
        for (int t = 0; t < StftFrames; t++) padded.Slice(t * HopSamples, FftSize).CopyTo(frames[(t * FftSize)..]);
        backend.Linear(_spectrum, _frames, _stftMatrix, null);
        Magnitude();

        _enc1.Run(backend, _magnitude);
        _enc2.Run(backend, _enc1.Activated);
        _enc3.Run(backend, _enc2.Activated);
        _enc4.Run(backend, _enc3.Activated);

        // LstmOps carries PyTorch's i,f,g,o gate order, which is how the safetensors stores the two matrices.
        backend.Linear(_gatesInput, _enc4.Activated, _weightIh, _biasIh);
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

    /// <summary>Collapses the 129 real and 129 imaginary DFT channels to their magnitude, frame by frame. There is
    /// no <c>IBackend</c> complex-magnitude op and the sweep is 516 floats, so it runs on the CPU pointers.</summary>
    private void Magnitude()
    {
        float* spec = (float*)_spectrum.DataPointer;
        float* mag = (float*)_magnitude.DataPointer;
        for (int t = 0; t < StftFrames; t++)
        {
            float* re = spec + t * 2 * SpecBins;
            float* im = re + SpecBins;
            float* row = mag + t * SpecBins;
            for (int bin = 0; bin < SpecBins; bin++) row[bin] = MathF.Sqrt(re[bin] * re[bin] + im[bin] * im[bin]);
        }
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
        _enc1.Dispose(); _enc2.Dispose(); _enc3.Dispose(); _enc4.Dispose();
        Tensor[] scratch =
        [
            _frames, _spectrum, _magnitude, _gatesInput, _gatesHidden, _gates, _headInput, _logit, _probability,
            _hidden, _cell, _hiddenNext, _cellNext,
        ];
        foreach (Tensor t in scratch) t.Dispose();
    }

    private readonly record struct ConvW(Tensor Weight, Tensor Bias);

    /// <summary>One encoder layer — kernel 3, padding 1, ReLU — as unfold then matrix product. Input and output
    /// are time-major; <see cref="Activated"/> is the output after the ReLU.</summary>
    private sealed class Encoder : IDisposable
    {
        private readonly int _tIn;
        private readonly int _tOut;
        private readonly int _channelsIn;
        private readonly int _channelsOut;
        private readonly int _stride;
        private readonly Tensor _columns;
        private readonly Tensor _output;
        private Tensor? _matrix;
        private Tensor? _bias;

        public Encoder(int tIn, int channelsIn, int channelsOut, int stride)
        {
            _tIn = tIn;
            _tOut = (tIn + 2 - Kernel) / stride + 1;
            _channelsIn = channelsIn;
            _channelsOut = channelsOut;
            _stride = stride;
            _columns = new Tensor(new TensorShape(_tOut, channelsIn * Kernel), DType.F32);
            _output = new Tensor(new TensorShape(_tOut, channelsOut), DType.F32);
            Activated = new Tensor(new TensorShape(_tOut, channelsOut), DType.F32);
        }

        public Tensor Activated { get; }

        public void Bind(ConvW conv)
        {
            _matrix = conv.Weight.Reshape(new TensorShape(_channelsOut, _channelsIn * Kernel));
            _bias = conv.Bias;
        }

        public void Run(IBackend backend, Tensor input)
        {
            ReadOnlySpan<float> x = input.AsSpan<float>();
            Span<float> columns = _columns.AsSpan<float>();
            int width = _channelsIn * Kernel;
            // Row t holds input steps t*stride-1 .. t*stride+1, at column ic*K + k to match the weight's flattening.
            for (int t = 0; t < _tOut; t++)
            {
                Span<float> row = columns.Slice(t * width, width);
                for (int k = 0; k < Kernel; k++)
                {
                    int source = t * _stride + k - 1;
                    if ((uint)source >= (uint)_tIn)
                    {
                        for (int ic = 0; ic < _channelsIn; ic++) row[ic * Kernel + k] = 0f;
                        continue;
                    }
                    ReadOnlySpan<float> step = x.Slice(source * _channelsIn, _channelsIn);
                    for (int ic = 0; ic < _channelsIn; ic++) row[ic * Kernel + k] = step[ic];
                }
            }
            backend.Linear(_output, _columns, _matrix!, _bias);
            backend.LeakyRelu(Activated, _output, 0f);
        }

        public void Dispose()
        {
            _columns.Dispose();
            _output.Dispose();
            Activated.Dispose();
        }
    }
}
