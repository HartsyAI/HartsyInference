namespace HartsyInference.Core.Models;

/// <summary>The file format of a loaded model.</summary>
public enum ModelFormat : byte
{
    /// <summary>HuggingFace safetensors format (.safetensors).</summary>
    SafeTensors = 0,

    /// <summary>GGML universal format (.gguf) — supports quantized weights.</summary>
    Gguf = 1,

    /// <summary>ONNX format (.onnx) — used as a fallback passthrough.</summary>
    Onnx = 2,
}
