using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.PyTorch;

/// <summary>One-time conversion of a PyTorch pickle checkpoint into a safetensors file.
///
/// <para>Several models ship weights only as <c>.pth</c>/<c>.bin</c>/<c>.th</c> pickles and are repacked on first use so the normal safetensors load path can serve every subsequent run. That repack is the same four steps every time — load, optionally filter/rename keys, verify non-empty, save — so it lives here once instead of being hand-rolled per model.</para>
///
/// <para><b>Mirror-vs-link policy</b> for deciding whether a model needs this at all:
/// <list type="number">
/// <item>Link the original upstream file directly when it is already safetensors (or a format the engine loads natively) and reasonably sized.</item>
/// <item>Prefer an existing trustworthy third-party safetensors re-upload with a verifiable hash over building our own repack.</item>
/// <item>Self-host a repack only when upstream is pickle-only, or when trimming to just the tensors actually needed is a real size win (X-Codec went 1.36GB → 0.8GB by dropping encoder-only tensors).</item>
/// <item>When self-hosting, always keep the original upstream URL wired as the download fallback so the mirror is never a hard dependency.</item></list></para></summary>
public static class PickleCheckpointRepacker
{
    /// <summary>Loads a pickle checkpoint and writes the selected tensors to <paramref name="outputPath"/> as safetensors, via a temp file so a crash mid-write can't leave a truncated checkpoint in place.</summary>
    /// <param name="keyMap">Maps each source key to its output key, or returns null to drop that tensor. Null keeps every tensor under its original key. Source order is preserved, which keeps the output byte-identical to the hand-rolled conversions this replaced.</param>
    /// <param name="recursiveFlatten">Passed through to <see cref="PytorchPickleLoader.Load"/> — true for checkpoints whose root is a dict of state dicts rather than a flat state dict.</param>
    /// <returns>The number of tensors written.</returns>
    public static int Repack(string sourcePath, string outputPath, Func<string, string?>? keyMap = null, bool recursiveFlatten = false)
    {
        using PytorchPickleLoader loader = new();
        loader.Load(sourcePath, recursiveFlatten);
        Dictionary<string, Tensor> keep = new(StringComparer.Ordinal);
        foreach ((string key, Tensor tensor) in loader.GetAllTensors())
        {
            string? mapped = keyMap is null ? key : keyMap(key);
            if (mapped is not null)
            {
                keep[mapped] = tensor;
            }
        }
        if (keep.Count == 0)
        {
            throw new InvalidDataException($"Checkpoint '{sourcePath}' yielded no usable tensors — unexpected key layout.");
        }
        string? outputDir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(outputDir))
        {
            Directory.CreateDirectory(outputDir);
        }
        string tempPath = outputPath + ".tmp";
        try
        {
            SafeTensorsWriter.Save(tempPath, keep);
            File.Move(tempPath, outputPath, overwrite: true);
        }
        catch
        {
            try
            {
                File.Delete(tempPath);
            }
            catch
            {
                // Best-effort cleanup; the original failure is what matters.
            }
            throw;
        }
        Logs.Info($"[Repack] Wrote {keep.Count} tensors from '{Path.GetFileName(sourcePath)}' to '{Path.GetFileName(outputPath)}'.");
        return keep.Count;
    }
}
