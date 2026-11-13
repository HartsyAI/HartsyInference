namespace HartsyInference.ModelAssets.PyTorch;

/// <summary>Where a pickle checkpoint's safetensors repack sits: beside it, under the same name with the extension <c>.safetensors</c> (<c>unet/diffusion_pytorch_model.bin</c> has <c>unet/diffusion_pytorch_model.safetensors</c>).
/// A repack holds the pickle's tensors under the same names, so a loader that prefers it reads the same weights without parsing a pickle.</summary>
public static class RepackedCheckpoint
{
    /// <summary>The extension of a repack.</summary>
    public const string Extension = ".safetensors";

    /// <summary>The path a repack of <paramref name="path"/> has. A path that is already safetensors names itself.</summary>
    public static string SiblingOf(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        return IsSafeTensorsName(path) ? path : Path.ChangeExtension(path, Extension);
    }

    /// <summary>The file a loader should read for <paramref name="path"/>: the repack when it exists, else <paramref name="path"/>.</summary>
    public static string Resolve(string path)
    {
        string sibling = SiblingOf(path);
        return File.Exists(sibling) ? sibling : path;
    }

    /// <summary>Whether <paramref name="path"/> or its repack is on disk.</summary>
    public static bool Exists(string path) => File.Exists(path) || File.Exists(SiblingOf(path));

    private static bool IsSafeTensorsName(string path) => path.EndsWith(Extension, StringComparison.OrdinalIgnoreCase);
}
