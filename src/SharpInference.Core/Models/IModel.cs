using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Core.Models;

/// <summary>Interface for a loaded model. Provides access to model weights, configuration, and a forward pass method. Model implementations are responsible for mapping loaded tensor weights to their internal architecture.</summary>
public interface IModel : IDisposable
{
    /// <summary>Configuration describing the model architecture and parameters.</summary>
    ModelConfig Config { get; }

    /// <summary>The device this model's weights are loaded on.</summary>
    DeviceKind Device { get; }

    /// <summary>Loads model weights from a dictionary of named tensors (as returned by a model loader).</summary>
    /// <param name="weights">Tensor name → tensor data mapping from safetensors/GGUF loader.</param>
    /// <param name="backend">Backend to use for any weight preprocessing (e.g., dtype conversion).</param>
    void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, IBackend backend);

    /// <summary>Runs the model's forward pass.</summary>
    /// <param name="output">Tensor to write the output into.</param>
    /// <param name="input">Input tensor(s) — meaning depends on model type.</param>
    /// <param name="backend">Backend to execute operations on.</param>
    void Forward(Tensor output, Tensor input, IBackend backend);
}
