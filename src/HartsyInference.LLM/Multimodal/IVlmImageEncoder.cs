using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Multimodal;

/// <summary>A VLM vision tower + connector: encodes a preprocessed image into a set of embeddings in the text
/// decoder's hidden space, ready to splice into the input sequence. Implemented by <see cref="SiglipVlmEncoder"/>
/// (SigLIP/CLIP families) and <see cref="Qwen25VlEncoder"/> (Qwen2.5-VL windowed 2D-RoPE ViT).</summary>
public interface IVlmImageEncoder : IDisposable
{
    /// <summary>Encodes a preprocessed image <c>[1, 3, H, W]</c> into <c>[1, tokensPerImage, textHidden]</c>.</summary>
    Tensor Encode(IBackend backend, Tensor pixelValues);

    /// <summary>Number of image embedding tokens this encoder emits for one image.</summary>
    int TokensPerImage { get; }

    /// <summary>Preferred square input size (pixels) for the synthetic test harness.</summary>
    int ImageSize { get; }

    float[] ImageMean { get; }
    float[] ImageStd { get; }

    /// <summary>Family tag used by <see cref="MultimodalGenerator"/> to pick the prompt format
    /// (<c>gemma3</c> | <c>idefics3</c> | <c>llava</c> | <c>qwen25vl</c>).</summary>
    string Family { get; }
}
