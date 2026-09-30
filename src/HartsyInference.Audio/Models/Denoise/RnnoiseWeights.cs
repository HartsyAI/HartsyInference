using HartsyInference.Audio.Models.Wake;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Denoise;

/// <summary>The 20 trained tensors of an RNNoise checkpoint, loaded once and shared by every stream.
///
/// <para>Split out from <see cref="RnnoiseModel"/> because the weights are stateless and 11.5 MB, while the
/// model around them is pure per-stream state (conv history, three GRU hidden vectors, scratch). A listener
/// serving dozens of satellites would otherwise hold dozens of identical copies. Same division the wake models
/// already use, where <c>WakeModelSet</c> shares one mel front-end and embedding across all sessions.</para>
///
/// <para>Borrowed by models, never owned: dispose this only after every <see cref="RnnoiseModel"/> built on it
/// is gone.</para></summary>
public sealed class RnnoiseWeights : IDisposable
{
    /// <summary>Tensors in a complete checkpoint: two convs, three GRUs of four, and two dense heads.</summary>
    public const int TensorCount = 20;

    private const string Source = "rnnoise.safetensors converted from xiph's rnnoise10Ga_12.pth (RnnoiseCheckpoint)";
    private const int Gates = 3 * RnnoiseModel.GruSize;
    private const int CatSize = 4 * RnnoiseModel.GruSize;

    private int _disposed;

    public Tensor Conv1Weight { get; private set; } = null!;
    public Tensor Conv1Bias { get; private set; } = null!;
    public Tensor Conv2Weight { get; private set; } = null!;
    public Tensor Conv2Bias { get; private set; } = null!;
    public Tensor DenseOutWeight { get; private set; } = null!;
    public Tensor DenseOutBias { get; private set; } = null!;
    public Tensor VadWeight { get; private set; } = null!;
    public Tensor VadBias { get; private set; } = null!;

    /// <summary>Per-GRU input-side weights, indexed 0-2 for gru1-gru3.</summary>
    public Tensor[] GruWeightIh { get; } = new Tensor[3];
    public Tensor[] GruWeightHh { get; } = new Tensor[3];
    public Tensor[] GruBiasIh { get; } = new Tensor[3];
    public Tensor[] GruBiasHh { get; } = new Tensor[3];

    /// <summary>True once <see cref="Load"/> has bound every tensor.</summary>
    public bool IsLoaded { get; private set; }

    /// <summary>Takes owned F32 copies via <see cref="WakeWeights"/>, so the loader that supplied them can be
    /// disposed immediately afterwards. Throws naming the tensor when one is missing or shaped for another
    /// architecture, which would otherwise surface as a shape error on the first speech frame.</summary>
    public void Load(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        Conv1Weight = Require(weights, "conv1.weight",
            new TensorShape(RnnoiseModel.CondSize, RnnoiseModel.InputDim, RnnoiseModel.KernelSize));
        Conv1Bias = Require(weights, "conv1.bias", new TensorShape(RnnoiseModel.CondSize));
        Conv2Weight = Require(weights, "conv2.weight",
            new TensorShape(RnnoiseModel.GruSize, RnnoiseModel.CondSize, RnnoiseModel.KernelSize));
        Conv2Bias = Require(weights, "conv2.bias", new TensorShape(RnnoiseModel.GruSize));
        for (int i = 0; i < 3; i++)
        {
            string gru = $"gru{i + 1}";
            GruWeightIh[i] = Require(weights, $"{gru}.weight_ih_l0", new TensorShape(Gates, RnnoiseModel.GruSize));
            GruWeightHh[i] = Require(weights, $"{gru}.weight_hh_l0", new TensorShape(Gates, RnnoiseModel.GruSize));
            GruBiasIh[i] = Require(weights, $"{gru}.bias_ih_l0", new TensorShape(Gates));
            GruBiasHh[i] = Require(weights, $"{gru}.bias_hh_l0", new TensorShape(Gates));
        }
        DenseOutWeight = Require(weights, "dense_out.weight", new TensorShape(RnnoiseModel.OutputDim, CatSize));
        DenseOutBias = Require(weights, "dense_out.bias", new TensorShape(RnnoiseModel.OutputDim));
        VadWeight = Require(weights, "vad_dense.weight", new TensorShape(1, CatSize));
        VadBias = Require(weights, "vad_dense.bias", new TensorShape(1));
        IsLoaded = true;
    }

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string name, TensorShape expected)
    {
        if (weights.TryGetValue(name, out Tensor? tensor) && tensor.Shape != expected)
            throw new HartsyInferenceException(
                $"Weight '{name}' has shape {tensor.Shape}, expected {expected}. Expected {Source}.");
        return WakeWeights.Require(weights, name, Source);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        // Null-conditional throughout: a Load that threw part-way has bound only some of these.
        Conv1Weight?.Dispose(); Conv1Bias?.Dispose(); Conv2Weight?.Dispose(); Conv2Bias?.Dispose();
        DenseOutWeight?.Dispose(); DenseOutBias?.Dispose(); VadWeight?.Dispose(); VadBias?.Dispose();
        for (int i = 0; i < 3; i++)
        {
            GruWeightIh[i]?.Dispose(); GruWeightHh[i]?.Dispose();
            GruBiasIh[i]?.Dispose(); GruBiasHh[i]?.Dispose();
        }
    }
}
