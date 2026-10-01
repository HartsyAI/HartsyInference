using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Wake;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Denoise;

/// <summary>RNNoise's recurrent network: 65 features in, 32 per-band gains plus a speech probability out.
///
/// <para>Two causal Conv1d layers (tanh) feed three stacked GRUs; the conv output and all three GRU outputs are
/// concatenated into 1536 values that two sigmoid heads read. The skip-concatenation is the point of the
/// architecture — the gain head sees both the immediate spectral context and three depths of temporal memory,
/// which is what lets it hold a gain steady through a syllable instead of chattering per frame.</para>
///
/// <para>Ported against the <b>PyTorch</b> definition (<c>torch/rnnoise/rnnoise.py</c>: stock
/// <c>nn.Conv1d</c> / <c>nn.GRU</c> / <c>nn.Linear</c>), not the C blob, because the weights come from the
/// distributed <c>.pth</c>. That makes PyTorch's <c>(r, z, n)</c> GRU gate order and reset-gate-on-hidden
/// convention the contract, which is exactly what <see cref="GruOps.GateAndUpdate"/> already implements — so
/// the gate math is reused rather than rewritten.</para>
///
/// <para>Every buffer is preallocated and reused: this runs 100 times a second per stream for the life of the
/// process, so a per-frame tensor allocation would be permanent native-heap churn. Same discipline, and the
/// same shape, as <c>SileroVad</c>.</para>
///
/// <para>Holds streaming state (conv history, three GRU hidden vectors) and <b>borrows</b> its
/// <see cref="RnnoiseWeights"/>; one instance per stream, not thread-safe. Disposing this leaves the weights
/// alone, so they outlive any number of streams built on them.</para>
///
/// <para>At <see cref="RnnoisePrecision.Int8"/>, conv2 and the GRUs run as upstream's default C build runs them: each
/// input is encoded to uint8 (<see cref="IBackend.QuantizeActivationsU8"/>) and multiplied by the int8 table
/// (<see cref="IBackend.LinearI8U8"/>) with the SU bias, and the recurrent products add their float diagonal on the F32
/// hidden state. conv1 and the two heads stay F32, as there. Everything else, the gate math included, is shared.</para></summary>
public sealed class RnnoiseModel : IDisposable
{
    /// <summary>Feature vector width — see <see cref="RnnoiseBands.FeatureCount"/>.</summary>
    public const int InputDim = 65;

    /// <summary>Channels out of the first conv.</summary>
    public const int CondSize = 128;

    /// <summary>Width of each GRU and of the second conv.</summary>
    public const int GruSize = 384;

    /// <summary>Band gains produced per frame.</summary>
    public const int OutputDim = 32;

    /// <summary>Conv kernel; 'valid' padding means each output frame needs two frames of history.</summary>
    public const int KernelSize = 3;

    private const int CatSize = 4 * GruSize;
    private const int Gates = 3 * GruSize;

    private readonly RnnoiseWeights _weights;

    private readonly Tensor _conv1Input = new(new TensorShape(1, InputDim, KernelSize), DType.F32);
    private readonly Tensor _conv1Out = new(new TensorShape(1, CondSize), DType.F32);
    private readonly Tensor _conv1Act = new(new TensorShape(1, CondSize), DType.F32);
    private readonly Tensor _conv2Input = new(new TensorShape(1, CondSize, KernelSize), DType.F32);
    private readonly Tensor _conv2Out = new(new TensorShape(1, GruSize), DType.F32);
    private readonly Tensor _conv2Act = new(new TensorShape(1, GruSize), DType.F32);
    private readonly Tensor _conv1Window;
    private readonly Tensor _conv2Window;
    private readonly Tensor _conv1Matrix;
    private readonly Tensor _conv2Matrix;
    private readonly Tensor _cat = new(new TensorShape(1, CatSize), DType.F32);
    private readonly Tensor _gruInput = new(new TensorShape(1, GruSize), DType.F32);
    private readonly Tensor _gi = new(new TensorShape(1, Gates), DType.F32);
    private readonly Tensor _gh = new(new TensorShape(1, Gates), DType.F32);
    private readonly Tensor[] _hidden;
    private readonly Tensor[] _hiddenNext;
    private readonly Tensor _gainLogits = new(new TensorShape(1, OutputDim), DType.F32);
    private readonly Tensor _gains = new(new TensorShape(1, OutputDim), DType.F32);
    private readonly Tensor _vadLogit = new(new TensorShape(1, 1), DType.F32);
    private readonly Tensor _vadOut = new(new TensorShape(1, 1), DType.F32);

    // Two frames at a time, one row each; see ProcessPair.
    private readonly Tensor _conv1Windows = new(new TensorShape(2, InputDim * KernelSize), DType.F32);
    private readonly Tensor _conv1Out2 = new(new TensorShape(2, CondSize), DType.F32);
    private readonly Tensor _conv1Act2 = new(new TensorShape(2, CondSize), DType.F32);
    private readonly Tensor _conv2Windows = new(new TensorShape(2, CondSize * KernelSize), DType.F32);
    private readonly Tensor _conv2Out2 = new(new TensorShape(2, GruSize), DType.F32);
    private readonly Tensor _conv2Act2 = new(new TensorShape(2, GruSize), DType.F32);
    private readonly Tensor _cat2 = new(new TensorShape(2, CatSize), DType.F32);
    private readonly Tensor _gruInput2 = new(new TensorShape(2, GruSize), DType.F32);
    private readonly Tensor _gi2 = new(new TensorShape(2, Gates), DType.F32);
    private readonly Tensor[] _gi2Rows;
    private readonly Tensor _gainLogits2 = new(new TensorShape(2, OutputDim), DType.F32);
    private readonly Tensor _gains2 = new(new TensorShape(2, OutputDim), DType.F32);
    private readonly Tensor _vadLogit2 = new(new TensorShape(2, 1), DType.F32);
    private readonly Tensor _vadOut2 = new(new TensorShape(2, 1), DType.F32);

    // uint8 codes of up to two rows of a product's input, at Int8 only. Every int8 product here takes 384 inputs.
    private readonly Tensor? _codes;
    private readonly Tensor? _codesRow;
    private int _disposed;

    /// <summary>Builds a stream over shared <paramref name="weights"/>, which must already be loaded and which
    /// this does not take ownership of.</summary>
    public RnnoiseModel(RnnoiseWeights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (!weights.IsLoaded) throw new InvalidOperationException("RnnoiseWeights have not been loaded.");
        _weights = weights;
        // Borrowed views, flattening each conv to the matrix-vector product it is at one output step.
        _conv1Window = _conv1Input.Reshape(new TensorShape(1, InputDim * KernelSize));
        _conv2Window = _conv2Input.Reshape(new TensorShape(1, CondSize * KernelSize));
        _conv1Matrix = weights.Conv1Weight.Reshape(new TensorShape(CondSize, InputDim * KernelSize));
        _conv2Matrix = weights.Conv2Weight.Reshape(new TensorShape(GruSize, CondSize * KernelSize));
        _hidden = [.. Enumerable.Range(0, 3).Select(_ => new Tensor(new TensorShape(1, GruSize), DType.F32))];
        _hiddenNext = [.. Enumerable.Range(0, 3).Select(_ => new Tensor(new TensorShape(1, GruSize), DType.F32))];
        _gi2Rows = [_gi2.SliceRows(0, 1), _gi2.SliceRows(1, 1)];
        if (weights.Precision == RnnoisePrecision.Int8)
        {
            _codes = new Tensor(new TensorShape(2, GruSize), DType.U8);
            _codesRow = _codes.SliceRows(0, 1);
        }
        Reset();
    }

    /// <summary>Runs one frame. <paramref name="features"/> is 65 values; <paramref name="bandGains"/> receives
    /// 32 sigmoid gains and <paramref name="speechProbability"/> the VAD head's output.</summary>
    public void Process(IBackend backend, ReadOnlySpan<float> features, Span<float> bandGains,
        out float speechProbability)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (features.Length < InputDim)
            throw new ArgumentException($"features must hold {InputDim} values.", nameof(features));
        if (bandGains.Length < OutputDim)
            throw new ArgumentException($"bandGains must hold {OutputDim} values.", nameof(bandGains));

        // Each conv emits one step from a three-frame window with no padding, which makes it a matrix-vector
        // product: the weight's [out, in, k] and the window's [in, k] flatten to the same in*K+k order. Routed
        // through Linear because the generic Conv1d kernel spends most of its time on per-tap bookkeeping at T=1.
        ShiftIn(_conv1Input.AsSpan<float>(), features, InputDim);
        backend.Linear(_conv1Out, _conv1Window, _conv1Matrix, _weights.Conv1Bias);
        backend.Tanh(_conv1Act, _conv1Out);

        ShiftIn(_conv2Input.AsSpan<float>(), _conv1Act.AsSpan<float>(), CondSize);
        Conv2(backend, _conv2Out, _conv2Window);
        backend.Tanh(_conv2Act, _conv2Out);

        Span<float> cat = _cat.AsSpan<float>();
        _conv2Act.AsSpan<float>().CopyTo(cat);

        Span<float> gruInput = _gruInput.AsSpan<float>();
        cat[..GruSize].CopyTo(gruInput);
        for (int layer = 0; layer < 3; layer++)
        {
            GruInputProjection(backend, layer, _gi, _gruInput);
            GruRecurrentProjection(backend, layer);
            GruOps.GateAndUpdate(_gi, _gh, _hidden[layer], _hiddenNext[layer], 1, GruSize);
            (_hidden[layer], _hiddenNext[layer]) = (_hiddenNext[layer], _hidden[layer]);
            Span<float> h = _hidden[layer].AsSpan<float>();
            h.CopyTo(cat.Slice((layer + 1) * GruSize, GruSize));
            h.CopyTo(gruInput);
        }

        backend.Linear(_gainLogits, _cat, _weights.DenseOutWeight, _weights.DenseOutBias);
        backend.Sigmoid(_gains, _gainLogits);
        backend.Linear(_vadLogit, _cat, _weights.VadWeight, _weights.VadBias);
        backend.Sigmoid(_vadOut, _vadLogit);

        _gains.AsSpan<float>()[..OutputDim].CopyTo(bandGains);
        speechProbability = _vadOut.AsSpan<float>()[0];
    }

    /// <summary>Runs two consecutive frames, <paramref name="first"/> then <paramref name="second"/>, with the same
    /// results and the same state afterwards as two <see cref="Process"/> calls, bit for bit.</summary>
    /// <remarks><para>What changes is how often each weight is read. One frame's forward reads every weight once,
    /// which is 11.5 MB of F32. When other cores keep that out of the cache, the frame costs that much DRAM traffic.
    /// Run layer by layer across the pair instead:</para>
    /// <list type="bullet">
    /// <item>both conv windows go through each conv in one product;</item>
    /// <item>each GRU's input projection <c>W·x</c> takes both frames' inputs in one pass over <c>W</c>;</item>
    /// <item>the two dense heads take both rows of the concatenation.</item>
    /// </list>
    /// <para>Only the recurrent products <c>U·h</c> stay one frame at a time, since frame two needs frame one's
    /// state. That reads 5.3 MB of <c>W</c>, 0.7 MB of conv and 0.2 MB of head weights once per pair instead of
    /// twice, about 16.8 MB in place of 23.1 MB. Every output element is the same sum in the same order as in
    /// <see cref="Process"/>: the products are row-independent, and the elementwise activations do not care which
    /// row an element sits in.</para></remarks>
    public void ProcessPair(IBackend backend, ReadOnlySpan<float> first, ReadOnlySpan<float> second,
        Span<float> firstGains, Span<float> secondGains, out float firstSpeech, out float secondSpeech)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (first.Length < InputDim || second.Length < InputDim)
            throw new ArgumentException($"features must hold {InputDim} values per frame.", nameof(first));
        if (firstGains.Length < OutputDim || secondGains.Length < OutputDim)
            throw new ArgumentException($"gains must hold {OutputDim} values per frame.", nameof(firstGains));

        // Each frame's window is the persistent window after that frame is shifted in, as Process would see it.
        Span<float> conv1State = _conv1Input.AsSpan<float>();
        Span<float> conv1Windows = _conv1Windows.AsSpan<float>();
        int conv1Width = InputDim * KernelSize;
        ShiftIn(conv1State, first, InputDim);
        conv1State.CopyTo(conv1Windows[..conv1Width]);
        ShiftIn(conv1State, second, InputDim);
        conv1State.CopyTo(conv1Windows[conv1Width..]);
        backend.Linear(_conv1Out2, _conv1Windows, _conv1Matrix, _weights.Conv1Bias);
        backend.Tanh(_conv1Act2, _conv1Out2);

        Span<float> conv1Act = _conv1Act2.AsSpan<float>();
        Span<float> conv2State = _conv2Input.AsSpan<float>();
        Span<float> conv2Windows = _conv2Windows.AsSpan<float>();
        int conv2Width = CondSize * KernelSize;
        ShiftIn(conv2State, conv1Act[..CondSize], CondSize);
        conv2State.CopyTo(conv2Windows[..conv2Width]);
        ShiftIn(conv2State, conv1Act[CondSize..], CondSize);
        conv2State.CopyTo(conv2Windows[conv2Width..]);
        Conv2(backend, _conv2Out2, _conv2Windows);
        backend.Tanh(_conv2Act2, _conv2Out2);

        Span<float> cat = _cat2.AsSpan<float>();
        Span<float> gruInput = _gruInput2.AsSpan<float>();
        Span<float> conv2Act = _conv2Act2.AsSpan<float>();
        for (int frame = 0; frame < 2; frame++)
        {
            ReadOnlySpan<float> row = conv2Act.Slice(frame * GruSize, GruSize);
            row.CopyTo(cat.Slice(frame * CatSize, GruSize));
            row.CopyTo(gruInput.Slice(frame * GruSize, GruSize));
        }
        for (int layer = 0; layer < 3; layer++)
        {
            // Both frames' input projections in one pass over W; the recurrence then runs a frame at a time.
            GruInputProjection(backend, layer, _gi2, _gruInput2);
            for (int frame = 0; frame < 2; frame++)
            {
                GruRecurrentProjection(backend, layer);
                GruOps.GateAndUpdate(_gi2Rows[frame], _gh, _hidden[layer], _hiddenNext[layer], 1, GruSize);
                (_hidden[layer], _hiddenNext[layer]) = (_hiddenNext[layer], _hidden[layer]);
                Span<float> h = _hidden[layer].AsSpan<float>();
                h.CopyTo(cat.Slice(frame * CatSize + (layer + 1) * GruSize, GruSize));
                // This layer's projections are already taken, so the row can carry the next layer's input.
                h.CopyTo(gruInput.Slice(frame * GruSize, GruSize));
            }
        }

        backend.Linear(_gainLogits2, _cat2, _weights.DenseOutWeight, _weights.DenseOutBias);
        backend.Sigmoid(_gains2, _gainLogits2);
        backend.Linear(_vadLogit2, _cat2, _weights.VadWeight, _weights.VadBias);
        backend.Sigmoid(_vadOut2, _vadLogit2);

        ReadOnlySpan<float> gains = _gains2.AsSpan<float>();
        gains[..OutputDim].CopyTo(firstGains);
        gains[OutputDim..(2 * OutputDim)].CopyTo(secondGains);
        ReadOnlySpan<float> speech = _vadOut2.AsSpan<float>();
        firstSpeech = speech[0];
        secondSpeech = speech[1];
    }

    /// <summary>conv2 on one or two flattened windows (<paramref name="windows"/>, one per row): F32, or the int8 table
    /// on their uint8 codes.</summary>
    private void Conv2(IBackend backend, Tensor output, Tensor windows)
    {
        if (_codes is null)
        {
            backend.Linear(output, windows, _conv2Matrix, _weights.Conv2Bias);
            return;
        }
        Tensor codes = CodesFor(windows);
        backend.QuantizeActivationsU8(codes, windows);
        backend.LinearI8U8(output, codes, _weights.Conv2Int8!, _weights.Conv2Scale!, _weights.Conv2Subias!);
    }

    /// <summary>A GRU's input projection <c>W·x + b</c> for one or two rows of <paramref name="input"/>.</summary>
    private void GruInputProjection(IBackend backend, int layer, Tensor output, Tensor input)
    {
        if (_codes is null)
        {
            backend.Linear(output, input, _weights.GruWeightIh[layer], _weights.GruBiasIh[layer]);
            return;
        }
        Tensor codes = CodesFor(input);
        backend.QuantizeActivationsU8(codes, input);
        backend.LinearI8U8(output, codes, _weights.GruInputInt8[layer]!, _weights.GruInputScale[layer]!,
            _weights.GruInputSubias[layer]!);
    }

    /// <summary>A GRU's recurrent projection <c>U·h + b</c> of its current hidden state, into <c>_gh</c>. At int8 the
    /// diagonal of <c>U</c> is applied in float to the F32 state, as upstream keeps it.</summary>
    /// <remarks>The hidden state's codes go into row 0 of the same buffer the layer's input codes used. In the paired
    /// path both frames' input projections are already taken by then, so nothing still needs those codes.</remarks>
    private void GruRecurrentProjection(IBackend backend, int layer)
    {
        Tensor hidden = _hidden[layer];
        if (_codes is null)
        {
            backend.Linear(_gh, hidden, _weights.GruWeightHh[layer], _weights.GruBiasHh[layer]);
            return;
        }
        backend.QuantizeActivationsU8(_codesRow!, hidden);
        backend.LinearI8U8(_gh, _codesRow!, _weights.GruRecurrentInt8[layer]!, _weights.GruRecurrentScale[layer]!,
            _weights.GruRecurrentSubias[layer]!, _weights.GruRecurrentDiag[layer], hidden);
    }

    /// <summary>The code buffer rows for an input of one or two rows.</summary>
    private Tensor CodesFor(Tensor input) => input.Shape[0] == 1 ? _codesRow! : _codes!;

    /// <summary>Slides a channels-first <c>[1, C, 3]</c> conv window one frame left and writes the newest frame
    /// into the last slot. Channels-first means each channel's three timesteps are contiguous, so the shift is
    /// per-channel rather than one block move.</summary>
    private static void ShiftIn(Span<float> window, ReadOnlySpan<float> newFrame, int channels)
    {
        for (int c = 0; c < channels; c++)
        {
            int b = c * KernelSize;
            window[b] = window[b + 1];
            window[b + 1] = window[b + 2];
            window[b + 2] = newFrame[c];
        }
    }

    /// <summary>Clears conv history and GRU hidden state. Required on a stream discontinuity: the GRUs carry
    /// seconds of context, and resuming across a gap conditions the gains on audio that never adjoined.</summary>
    public void Reset()
    {
        _conv1Input.AsSpan<float>().Clear();
        _conv2Input.AsSpan<float>().Clear();
        for (int i = 0; i < 3; i++)
        {
            _hidden[i].AsSpan<float>().Clear();
            _hiddenNext[i].AsSpan<float>().Clear();
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor? t in EnumerateOwned()) t?.Dispose();
    }

    /// <summary>Only the per-stream scratch and state; the weights are borrowed and outlive this instance.</summary>
    private IEnumerable<Tensor?> EnumerateOwned()
    {
        for (int i = 0; i < 3; i++)
        {
            yield return _hidden[i]; yield return _hiddenNext[i];
        }
        yield return _conv1Input; yield return _conv1Out; yield return _conv1Act;
        yield return _conv2Input; yield return _conv2Out; yield return _conv2Act;
        yield return _cat; yield return _gruInput; yield return _gi; yield return _gh;
        yield return _gainLogits; yield return _gains; yield return _vadLogit; yield return _vadOut;
        yield return _conv1Windows; yield return _conv1Out2; yield return _conv1Act2;
        yield return _conv2Windows; yield return _conv2Out2; yield return _conv2Act2;
        yield return _cat2; yield return _gruInput2; yield return _gi2;
        yield return _gainLogits2; yield return _gains2; yield return _vadLogit2; yield return _vadOut2;
        yield return _codes;
    }
}
