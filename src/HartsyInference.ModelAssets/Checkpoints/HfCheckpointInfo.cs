using HartsyInference.ModelAssets.Quant;

namespace HartsyInference.ModelAssets.Checkpoints;

/// <summary>What a Hugging Face style checkpoint directory declares, read from <c>config.json</c> and the file listing alone.</summary>
/// <param name="Root">Full path of the directory.</param>
/// <param name="ConfigPath">Full path of <c>config.json</c>.</param>
/// <param name="IndexPath">Full path of <c>model.safetensors.index.json</c>, or null for a directory without a shard index.</param>
/// <param name="TokenizerPath">Full path of <c>tokenizer.json</c>, or null when the directory ships none.</param>
/// <param name="ModelType">The config's top-level <c>model_type</c>.</param>
/// <param name="Flavor">Which producer's quantization naming the weights follow, or null when the config declares no quantization.</param>
public sealed record HfCheckpointInfo(
    string Root,
    string ConfigPath,
    string? IndexPath,
    string? TokenizerPath,
    string ModelType,
    QuantFlavor? Flavor);
