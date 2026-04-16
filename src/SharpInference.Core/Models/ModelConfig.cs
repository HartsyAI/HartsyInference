namespace SharpInference.Core.Models;

/// <summary>Architecture parameters read from model metadata (safetensors config.json or GGUF metadata). Used by <see cref="IModel"/> implementations and pipeline factory to configure the correct pipeline.</summary>
public record ModelConfig
{
    /// <summary>Architecture identifier (e.g., "stable-diffusion-v1-5", "sdxl", "flux-dev").</summary>
    public required string Architecture { get; init; }

    /// <summary>Model format on disk.</summary>
    public required ModelFormat Format { get; init; }

    /// <summary>Data type of the model weights.</summary>
    public required Tensors.DType WeightDType { get; init; }

    /// <summary>Number of hidden dimensions in the primary model component.</summary>
    public int HiddenSize { get; init; }

    /// <summary>Number of attention heads.</summary>
    public int NumAttentionHeads { get; init; }

    /// <summary>Number of layers/blocks in the primary model component.</summary>
    public int NumLayers { get; init; }

    /// <summary>Vocabulary size for text models.</summary>
    public int VocabSize { get; init; }

    /// <summary>Maximum sequence length for text models.</summary>
    public int MaxSequenceLength { get; init; }

    /// <summary>VAE scaling factor (e.g., 0.18215 for SD1.5, 0.13025 for SDXL).</summary>
    public float VaeScaleFactor { get; init; }

    /// <summary>Prediction type: "epsilon", "v_prediction", "sample".</summary>
    public string PredictionType { get; init; } = "epsilon";

    /// <summary>Additional key-value metadata from the model file.</summary>
    public Dictionary<string, string> Metadata { get; init; } = [];
}
