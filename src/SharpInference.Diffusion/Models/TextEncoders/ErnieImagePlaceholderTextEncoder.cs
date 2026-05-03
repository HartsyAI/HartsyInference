using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.TextEncoders;

/// <summary>Placeholder <see cref="IErnieTextEncoder"/> that throws on <see cref="Encode"/>. Allows the pipeline to compile and be wired up in tests without a real encoder; replace with <see cref="ErnieImageLlamaTextEncoder"/> or a custom encoder once the Baidu <c>text_encoder/config.json</c> architecture is confirmed.</summary>
public sealed class ErnieImagePlaceholderTextEncoder : IErnieTextEncoder
{
    private readonly int _outputDim;

    /// <summary>Creates the placeholder. <paramref name="outputDim"/> is reported via <see cref="OutputDim"/> so the pipeline can size its <c>text_proj</c> input correctly even before a real encoder is available.</summary>
    public ErnieImagePlaceholderTextEncoder(int outputDim) => _outputDim = outputDim;

    /// <summary>Output feature dim (typically 3072 for ERNIE-Image v1).</summary>
    public int OutputDim => _outputDim;

    /// <summary>Always throws — plug in a real <see cref="IErnieTextEncoder"/> implementation before generating.</summary>
    public (Tensor HiddenStates, int[] TextLens) Encode(IBackend backend, int[][] tokenIds, int[] realLens) =>
        throw new NotSupportedException(
            "ErnieImagePlaceholderTextEncoder.Encode was called. Plug in a real IErnieTextEncoder " +
            "implementation (e.g., ErnieImageLlamaTextEncoder) before generating with ErnieImagePipeline.");

    /// <summary>No weights — placeholder.</summary>
    public IEnumerable<Tensor> EnumerateWeights() => [];

    /// <summary>No-op.</summary>
    public void Dispose() { }
}
