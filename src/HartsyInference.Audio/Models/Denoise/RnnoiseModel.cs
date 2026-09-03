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
/// alone, so they outlive any number of streams built on them.</para></summary>
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
    private readonly Tensor _conv1Out = new(new TensorShape(1, CondSize, 1), DType.F32);
    private readonly Tensor _conv1Act = new(new TensorShape(1, CondSize, 1), DType.F32);
    private readonly Tensor _conv2Input = new(new TensorShape(1, CondSize, KernelSize), DType.F32);
    private readonly Tensor _conv2Out = new(new TensorShape(1, GruSize, 1), DType.F32);
    private readonly Tensor _conv2Act = new(new TensorShape(1, GruSize, 1), DType.F32);
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
    private int _disposed;

    /// <summary>Builds a stream over shared <paramref name="weights"/>, which must already be loaded and which
    /// this does not take ownership of.</summary>
    public RnnoiseModel(RnnoiseWeights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (!weights.IsLoaded) throw new InvalidOperationException("RnnoiseWeights have not been loaded.");
        _weights = weights;
        _hidden = [.. Enumerable.Range(0, 3).Select(_ => new Tensor(new TensorShape(1, GruSize), DType.F32))];
        _hiddenNext = [.. Enumerable.Range(0, 3).Select(_ => new Tensor(new TensorShape(1, GruSize), DType.F32))];
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

        ShiftIn(_conv1Input.AsSpan<float>(), features, InputDim);
        backend.Conv1d(_conv1Out, _conv1Input, _weights.Conv1Weight, _weights.Conv1Bias, 1, 0, 0, 1, 1);
        backend.Tanh(_conv1Act, _conv1Out);

        ShiftIn(_conv2Input.AsSpan<float>(), _conv1Act.AsSpan<float>(), CondSize);
        backend.Conv1d(_conv2Out, _conv2Input, _weights.Conv2Weight, _weights.Conv2Bias, 1, 0, 0, 1, 1);
        backend.Tanh(_conv2Act, _conv2Out);

        Span<float> cat = _cat.AsSpan<float>();
        _conv2Act.AsSpan<float>().CopyTo(cat);

        Span<float> gruInput = _gruInput.AsSpan<float>();
        cat[..GruSize].CopyTo(gruInput);
        for (int layer = 0; layer < 3; layer++)
        {
            backend.Linear(_gi, _gruInput, _weights.GruWeightIh[layer], _weights.GruBiasIh[layer]);
            backend.Linear(_gh, _hidden[layer], _weights.GruWeightHh[layer], _weights.GruBiasHh[layer]);
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
    }
}
