namespace SharpInference.Diffusion.Pipelines;

/// <summary>Resolves file paths for all model components. Supports ComfyUI-style directory layouts.</summary>
public sealed record ModelPaths
{
    /// <summary>Path to the tokenizer directory containing vocab.json and merges.txt.</summary>
    public required string TokenizerDir { get; init; }

    /// <summary>Path to the text encoder safetensors file.</summary>
    public required string TextEncoderPath { get; init; }

    /// <summary>Path to the UNet safetensors file.</summary>
    public required string UNetPath { get; init; }

    /// <summary>Path to the VAE safetensors file.</summary>
    public required string VaePath { get; init; }

    /// <summary>Convenience property for vocab.json inside the tokenizer directory.</summary>
    public string VocabPath => Path.Combine(TokenizerDir, "vocab.json");

    /// <summary>Convenience property for merges.txt inside the tokenizer directory.</summary>
    public string MergesPath => Path.Combine(TokenizerDir, "merges.txt");

    /// <summary>Creates ModelPaths from a single model root with standard HuggingFace layout (tokenizer/, text_encoder/, unet/, vae/).</summary>
    public static ModelPaths FromHuggingFaceDir(string modelRoot, string weightFileName = "model.fp16.safetensors")
    {
        string root = Path.GetFullPath(modelRoot);
        return new ModelPaths
        {
            TokenizerDir = Path.Combine(root, "tokenizer"),
            TextEncoderPath = Path.Combine(root, "text_encoder", weightFileName),
            UNetPath = Path.Combine(root, "unet", $"diffusion_pytorch_model.fp16.safetensors"),
            VaePath = Path.Combine(root, "vae", $"diffusion_pytorch_model.fp16.safetensors"),
        };
    }

    /// <summary>Creates ModelPaths from a ComfyUI-style layout where each component type has its own top-level directory.</summary>
    public static ModelPaths FromComfyLayout(
        string modelsRoot,
        string checkpointName,
        string? vaeName = null,
        string tokenizerDir = "tokenizer")
    {
        string root = Path.GetFullPath(modelsRoot);
        string sdDir = Path.Combine(root, "Stable-Diffusion", checkpointName);

        return new ModelPaths
        {
            TokenizerDir = Path.Combine(sdDir, tokenizerDir),
            TextEncoderPath = Path.Combine(sdDir, "text_encoder", "model.fp16.safetensors"),
            UNetPath = Path.Combine(sdDir, "unet", "diffusion_pytorch_model.fp16.safetensors"),
            VaePath = vaeName != null
                ? Path.Combine(root, "VAE", vaeName)
                : Path.Combine(sdDir, "vae", "diffusion_pytorch_model.fp16.safetensors"),
        };
    }

    /// <summary>Validates that all required files exist. Throws FileNotFoundException for the first missing file.</summary>
    public void Validate()
    {
        string[] paths = [VocabPath, MergesPath, TextEncoderPath, UNetPath, VaePath];
        foreach (string path in paths)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"Required model file not found: {path}", path);
        }
    }
}

/// <summary>Resolves file paths for SDXL model components. Extends ModelPaths with dual tokenizer and dual text encoder support for CLIP-L + CLIP-G.</summary>
public sealed record SdxlModelPaths
{
    /// <summary>Path to the CLIP-L tokenizer directory (tokenizer/).</summary>
    public required string TokenizerLDir { get; init; }

    /// <summary>Path to the CLIP-G tokenizer directory (tokenizer_2/).</summary>
    public required string TokenizerGDir { get; init; }

    /// <summary>Path to the CLIP-L text encoder safetensors file (text_encoder/).</summary>
    public required string TextEncoderLPath { get; init; }

    /// <summary>Path to the CLIP-G text encoder safetensors file (text_encoder_2/).</summary>
    public required string TextEncoderGPath { get; init; }

    /// <summary>Path to the UNet safetensors file.</summary>
    public required string UNetPath { get; init; }

    /// <summary>Path to the VAE safetensors file.</summary>
    public required string VaePath { get; init; }

    /// <summary>CLIP-L vocab.json path.</summary>
    public string VocabLPath => Path.Combine(TokenizerLDir, "vocab.json");

    /// <summary>CLIP-L merges.txt path.</summary>
    public string MergesLPath => Path.Combine(TokenizerLDir, "merges.txt");

    /// <summary>CLIP-G vocab.json path.</summary>
    public string VocabGPath => Path.Combine(TokenizerGDir, "vocab.json");

    /// <summary>CLIP-G merges.txt path.</summary>
    public string MergesGPath => Path.Combine(TokenizerGDir, "merges.txt");

    /// <summary>Creates SdxlModelPaths from a HuggingFace diffusers layout (tokenizer/, tokenizer_2/, text_encoder/, text_encoder_2/, unet/, vae/).</summary>
    public static SdxlModelPaths FromHuggingFaceDir(string modelRoot, string weightFileName = "model.fp16.safetensors")
    {
        string root = Path.GetFullPath(modelRoot);
        return new SdxlModelPaths
        {
            TokenizerLDir = Path.Combine(root, "tokenizer"),
            TokenizerGDir = Path.Combine(root, "tokenizer_2"),
            TextEncoderLPath = Path.Combine(root, "text_encoder", weightFileName),
            TextEncoderGPath = Path.Combine(root, "text_encoder_2", weightFileName),
            UNetPath = Path.Combine(root, "unet", "diffusion_pytorch_model.fp16.safetensors"),
            VaePath = Path.Combine(root, "vae", "diffusion_pytorch_model.fp16.safetensors"),
        };
    }

    /// <summary>Validates that all required files exist. Throws FileNotFoundException for the first missing file.</summary>
    public void Validate()
    {
        string[] paths = [VocabLPath, MergesLPath, VocabGPath, MergesGPath, TextEncoderLPath, TextEncoderGPath, UNetPath, VaePath];
        foreach (string path in paths)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"Required model file not found: {path}", path);
        }
    }
}
