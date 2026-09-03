using HartsyInference.Audio.Models.Wake;
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
    private const string Source = "rnnoise.safetensors (converted from xiph/rnnoise rnnoise10Ga_*.pth)";

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
    /// disposed immediately afterwards.</summary>
    public void Load(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        Conv1Weight = WakeWeights.Require(weights, "conv1.weight", Source);
        Conv1Bias = WakeWeights.Require(weights, "conv1.bias", Source);
        Conv2Weight = WakeWeights.Require(weights, "conv2.weight", Source);
        Conv2Bias = WakeWeights.Require(weights, "conv2.bias", Source);
        for (int i = 0; i < 3; i++)
        {
            string gru = $"gru{i + 1}";
            GruWeightIh[i] = WakeWeights.Require(weights, $"{gru}.weight_ih_l0", Source);
            GruWeightHh[i] = WakeWeights.Require(weights, $"{gru}.weight_hh_l0", Source);
            GruBiasIh[i] = WakeWeights.Require(weights, $"{gru}.bias_ih_l0", Source);
            GruBiasHh[i] = WakeWeights.Require(weights, $"{gru}.bias_hh_l0", Source);
        }
        DenseOutWeight = WakeWeights.Require(weights, "dense_out.weight", Source);
        DenseOutBias = WakeWeights.Require(weights, "dense_out.bias", Source);
        VadWeight = WakeWeights.Require(weights, "vad_dense.weight", Source);
        VadBias = WakeWeights.Require(weights, "vad_dense.bias", Source);
        IsLoaded = true;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        if (!IsLoaded) return;
        Conv1Weight.Dispose(); Conv1Bias.Dispose(); Conv2Weight.Dispose(); Conv2Bias.Dispose();
        DenseOutWeight.Dispose(); DenseOutBias.Dispose(); VadWeight.Dispose(); VadBias.Dispose();
        for (int i = 0; i < 3; i++)
        {
            GruWeightIh[i].Dispose(); GruWeightHh[i].Dispose();
            GruBiasIh[i].Dispose(); GruBiasHh[i].Dispose();
        }
    }
}
