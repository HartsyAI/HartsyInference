using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Weight lookup shared by <see cref="MiniMaxH3VideoVaeDecoder"/> and its blocks. The checkpoint is all F16;
/// GEMM weights stay F16 (the Linear kernels cast them), but norm scales and biases are read as raw <c>float*</c> by
/// the norm kernels and must be promoted. Cast copies are appended to <c>owned</c> so the decoder can free them.</summary>
internal static class MiniMaxH3VideoVaeWeights
{
    private const string Family = "MiniMax-H3 video VAE";

    /// <summary>Weight as stored, naming the key when it is absent.</summary>
    public static Tensor Raw(IReadOnlyDictionary<string, Tensor> weights, string key) => CheckpointTensors.Require(weights, Family, key);

    /// <summary>F32 form of a required weight.</summary>
    public static Tensor F32(IReadOnlyDictionary<string, Tensor> weights, string key, List<Tensor> owned) =>
        CheckpointTensors.RequireF32(weights, Family, key, owned);

    /// <summary>F32 form of an optional weight (biases are optional in the reference's <c>bias=False</c> variants).</summary>
    public static Tensor? OptionalF32(IReadOnlyDictionary<string, Tensor> weights, string key, List<Tensor> owned) =>
        CheckpointTensors.OptionalF32(weights, Family, key, owned);
}
