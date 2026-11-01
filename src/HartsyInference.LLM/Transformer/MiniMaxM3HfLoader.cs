using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.LLM.Transformer;

/// <summary>The text decoder of a MiniMax-M3 Hugging Face directory (<c>config.json</c> plus safetensors shards) as the weights <see cref="GenericTransformer"/> reads,
/// with the MXFP8 attention and NVFP4 expert weights dequantized through <see cref="HfQuantCheckpoint"/>.</summary>
/// <remarks>The vision tower, projectors and the speculative <c>dspark/</c> head are not read. Every matrix is materialized in the requested dtype, so the real 250 GB checkpoint needs a host or placement plan sized for it.
/// A weight stored in that dtype is a view of the mapped shards, so this object owns them until it is disposed.</remarks>
public sealed class MiniMaxM3HfLoader : IDisposable
{
    private const string TextPrefix = "language_model.";
    private readonly HfQuantCheckpoint _checkpoint;

    private MiniMaxM3HfLoader(HfQuantCheckpoint checkpoint, TransformerConfig config, Dictionary<string, Tensor> weights)
    {
        _checkpoint = checkpoint;
        Config = config;
        Weights = weights;
    }

    public TransformerConfig Config { get; }

    /// <summary>The decoder weights by the names <see cref="GenericTransformer.LoadWeights"/> reads under the <c>model</c> prefix.</summary>
    public Dictionary<string, Tensor> Weights { get; }

    /// <summary>Reads <paramref name="directory"/>'s config and weights; matrices are produced as <paramref name="weightType"/>, norms, the router and its bias as F32.</summary>
    /// <exception cref="HartsyInferenceException">The directory is not a supported MiniMax-M3 checkpoint.</exception>
    public static MiniMaxM3HfLoader Load(string directory, DType weightType)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "config.json")));
        TransformerConfig config = MiniMaxM3HfConfigReader.FromHuggingFace(document.RootElement);
        HfQuantCheckpoint checkpoint = HfQuantCheckpoint.Open(directory);
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal);
        try
        {
            foreach ((string name, HfQuantWeight weight) in checkpoint.Weights)
            {
                if (!name.StartsWith(TextPrefix, StringComparison.Ordinal)) continue;
                string engineName = EngineName(name[TextPrefix.Length..]);
                bool matrix = weight.Shape.Rank == 2 && !engineName.EndsWith("mlp.gate.weight", StringComparison.Ordinal);
                weights[engineName] = checkpoint.Produce(name, matrix ? weightType : DType.F32);
            }
        }
        catch
        {
            foreach (Tensor tensor in weights.Values) tensor.Dispose();
            checkpoint.Dispose();
            throw;
        }
        return new MiniMaxM3HfLoader(checkpoint, config, weights);
    }

    public void Dispose()
    {
        foreach (Tensor tensor in Weights.Values) tensor.Dispose();
        Weights.Clear();
        _checkpoint.Dispose();
    }

    /// <summary>The name <see cref="GenericTransformer"/> reads for the checkpoint's <paramref name="name"/> (without the <c>language_model.</c> prefix).</summary>
    internal static string EngineName(string name)
    {
        string mapped = name
            .Replace(".block_sparse_moe.e_score_correction_bias", ".mlp.gate.e_score_correction_bias", StringComparison.Ordinal)
            .Replace(".block_sparse_moe.gate.", ".mlp.gate.", StringComparison.Ordinal)
            .Replace(".block_sparse_moe.shared_experts.", ".mlp.shared_expert.", StringComparison.Ordinal)
            .Replace(".block_sparse_moe.experts.", ".mlp.experts.", StringComparison.Ordinal)
            .Replace(".self_attn.index_", ".self_attn.indexer.", StringComparison.Ordinal);
        return mapped.Replace(".w1.", ".gate_proj.", StringComparison.Ordinal)
            .Replace(".w3.", ".up_proj.", StringComparison.Ordinal)
            .Replace(".w2.", ".down_proj.", StringComparison.Ordinal);
    }
}
