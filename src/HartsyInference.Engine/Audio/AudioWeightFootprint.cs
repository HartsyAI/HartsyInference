using System.Text.Json;
using HartsyInference.Audio.Cache;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Audio;

/// <summary>Sizes an audio model's weights from the files already on disk, for the switch check that decides whether
/// the model fits before it loads.</summary>
/// <remarks>Local and synchronous on purpose: the check runs under the generation lock and the device gate, so it never
/// asks a descriptor's <c>ResolveFiles</c> (which can probe the hub over HTTP) and never downloads. A model not fetched
/// yet sizes as 0, and every weight file in its folder counts, so a folder holding two variants sizes as both.</remarks>
internal static class AudioWeightFootprint
{
    /// <summary>Deep enough for a repo's component subfolders, shallow enough that a link loop cannot run away.</summary>
    private const int MaxDepth = 4;

    /// <summary>Extensions loaders read weights from; tokenizers, configs and manifests are noise beside them.</summary>
    private static readonly string[] WeightExtensions =
        [".safetensors", ".bin", ".pth", ".pt", ".ckpt", ".th", ".onnx", ".gguf"];

    /// <summary>Device bytes of the weights at <paramref name="source"/> — a local file or folder, or a HuggingFace repo
    /// id resolved to its cache folder under <paramref name="category"/> — or 0 when nothing is on disk.</summary>
    /// <param name="promotesHalfToF32">The runner widens F16/BF16 weights to F32 when it loads them, so those tensors
    /// count four bytes per element. Read from safetensors headers; any other format counts as stored.</param>
    internal static long Estimate(string? source, string category, bool promotesHalfToF32 = false)
    {
        if (string.IsNullOrWhiteSpace(source))
        {
            return 0;
        }
        try
        {
            if (File.Exists(source))
            {
                return FileBytes(source, promotesHalfToF32);
            }
            if (Path.IsPathRooted(source))
            {
                return Directory.Exists(source) ? DirectoryBytes(source, promotesHalfToF32) : 0;
            }
            return IsRepoId(source)
                ? DirectoryBytes(AudioModelCache.GetRepoDirectory(source, category), promotesHalfToF32) : 0;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            Logs.Debug($"[Audio] Could not size the weights at '{source}' ({ex.Message}); the switch check uses its floor.");
            return 0;
        }
    }

    /// <summary>Summed <see cref="FileBytes"/> of every weight file under <paramref name="directory"/>.</summary>
    internal static long DirectoryBytes(string directory, bool promotesHalfToF32)
    {
        EnumerationOptions options = new() { RecurseSubdirectories = true, MaxRecursionDepth = MaxDepth, IgnoreInaccessible = true };
        long total = 0;
        foreach (string path in Directory.EnumerateFiles(directory, "*", options))
        {
            if (WeightExtensions.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase))
            {
                total += FileBytes(path, promotesHalfToF32);
            }
        }
        return total;
    }

    /// <summary>The device bytes one weight file loads as.</summary>
    internal static long FileBytes(string path, bool promotesHalfToF32)
    {
        FileInfo info = new(path);
        // A symbolic stand-in reports the link's own length; a hard link already reports the file's.
        if (info.LinkTarget is not null && info.ResolveLinkTarget(returnFinalTarget: true) is FileInfo target)
        {
            info = target;
        }
        if (!info.Exists)
        {
            return 0;
        }
        return promotesHalfToF32 && TryPromotedBytes(info.FullName, out long promoted) ? promoted : info.Length;
    }

    /// <summary>A safetensors file's bytes with every F16/BF16 tensor widened to F32; false for any other format.</summary>
    private static bool TryPromotedBytes(string path, out long bytes)
    {
        bytes = 0;
        try
        {
            // Sniffed, not judged by extension: converted stand-ins keep the upstream name (pytorch_model.bin).
            if (!AnyFormatCheckpointLoader.IsSafeTensors(path))
            {
                return false;
            }
            foreach (SafeTensorDescriptor tensor in SafeTensorHeaderReader.Read(path).Tensors.Values)
            {
                bytes += tensor.DType == DType.F16 || tensor.DType == DType.BF16
                    ? tensor.Shape.ElementCount * sizeof(float) : tensor.ByteLength;
            }
            return true;
        }
        catch (Exception ex) when (ex is HartsyInferenceException or JsonException)
        {
            Logs.Debug($"[Audio] Unreadable safetensors header in '{path}' ({ex.Message}); sizing it as stored.");
            return false;
        }
    }

    /// <summary>A bare <c>owner/name</c> hub id — a local path or a composite cache key has no folder to size.</summary>
    private static bool IsRepoId(string value)
    {
        int slash = value.IndexOf('/');
        return slash > 0 && slash == value.LastIndexOf('/') && slash < value.Length - 1
            && value.IndexOfAny(['\\', '|', ':']) < 0;
    }
}
