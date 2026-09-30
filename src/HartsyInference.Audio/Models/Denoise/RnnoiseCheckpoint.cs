using System.Formats.Tar;
using System.IO.Compression;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Audio.Models.Denoise;

/// <summary>Converts xiph's RNNoise model tarball into the <c>rnnoise.safetensors</c> that
/// <see cref="RnnoiseWeights"/> loads.
///
/// <para>Upstream ships the weights only inside <c>rnnoise_data-{sha256}.tar.gz</c>, the file its
/// <c>download_model.sh</c> fetches. That holds two float PyTorch checkpoints and the C tables exported from them.
/// <see cref="CheckpointMember"/> is the checkpoint behind the library's default tables (<c>src/rnnoise_data.c</c>);
/// <c>rnnoise10Gb_15.pth</c> backs the smaller <c>rnnoise_data_little.c</c> and is not used. The checkpoint is
/// converted rather than the C tables because the tables carry int8 copies of conv2 and the GRUs.</para>
///
/// <para>The pickle goes through the safe-subset loader, never executed, and the converted file is loaded back
/// into <see cref="RnnoiseWeights"/> before it replaces anything, so a checkpoint of another architecture fails here
/// instead of on the first speech frame.</para></summary>
public static class RnnoiseCheckpoint
{
    /// <summary>Path, inside the tarball, of the checkpoint the default C build compiles in.</summary>
    public const string CheckpointMember = "models/rnnoise10Ga_12.pth";

    /// <summary>SPDX id of xiph/rnnoise's license (<c>COPYING</c>), which covers the model data.</summary>
    public const string License = "BSD-3-Clause";

    /// <summary>Converts the tarball's <see cref="CheckpointMember"/> to <paramref name="outputPath"/>.</summary>
    /// <param name="metadata">Written to the output header's <c>__metadata__</c>; provenance belongs here.</param>
    public static void ConvertTarball(string tarballPath, string outputPath,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(tarballPath);
        ArgumentException.ThrowIfNullOrEmpty(outputPath);
        string directory = Path.GetDirectoryName(Path.GetFullPath(outputPath))!;
        Directory.CreateDirectory(directory);
        // Beside the output, so a failed install leaves nothing behind in a shared temp directory.
        string checkpoint = Path.Combine(directory, $".{Path.GetFileName(outputPath)}.{Guid.NewGuid():N}.pth");
        try
        {
            ExtractMember(tarballPath, CheckpointMember, checkpoint);
            ConvertCheckpoint(checkpoint, outputPath, metadata);
        }
        finally
        {
            File.Delete(checkpoint);
        }
    }

    /// <summary>Converts an RNNoise checkpoint already on disk to <paramref name="outputPath"/>. An existing output is
    /// replaced only once the new file has loaded.</summary>
    /// <param name="metadata">Written to the output header's <c>__metadata__</c>.</param>
    public static void ConvertCheckpoint(string checkpointPath, string outputPath,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(checkpointPath);
        ArgumentException.ThrowIfNullOrEmpty(outputPath);
        // Unique per call, so two conversions to the same output cannot overwrite each other's staging file.
        string staging = $"{outputPath}.{Guid.NewGuid():N}.staging";
        try
        {
            // The .pth wraps its state dict beside model_args/model_kwargs/loss/epoch; the default load descends
            // into the one wrapper that holds tensors, so keys come out as conv1.weight rather than state_dict.*.
            PickleCheckpointRepacker.Repack(checkpointPath, staging, metadata: metadata);
            Validate(staging);
            File.Move(staging, outputPath, overwrite: true);
        }
        finally
        {
            File.Delete(staging);
        }
    }

    /// <summary>Copies the regular file <paramref name="member"/> out of a gzip-compressed tar.</summary>
    /// <exception cref="InvalidDataException">The archive has no such file.</exception>
    public static void ExtractMember(string tarballPath, string member, string destinationPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(member);
        using FileStream file = File.OpenRead(tarballPath);
        using GZipStream gzip = new(file, CompressionMode.Decompress);
        using TarReader reader = new(gzip);
        while (reader.GetNextEntry() is TarEntry entry)
        {
            bool regular = entry.EntryType is TarEntryType.RegularFile or TarEntryType.V7RegularFile;
            if (!regular || entry.DataStream is null || !NameMatches(entry.Name, member))
                continue;
            using FileStream output = new(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None);
            entry.DataStream.CopyTo(output);
            return;
        }
        throw new InvalidDataException($"'{tarballPath}' has no '{member}'; it is not xiph's RNNoise model tarball.");
    }

    private static bool NameMatches(string name, string member) =>
        string.Equals(name, member, StringComparison.Ordinal)
        || string.Equals(name, "./" + member, StringComparison.Ordinal);

    private static void Validate(string path)
    {
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> tensors = loader.GetAllTensors();
        try
        {
            if (tensors.Count != RnnoiseWeights.TensorCount)
                throw new InvalidDataException(
                    $"'{path}' holds {tensors.Count} tensors, expected RNNoise's {RnnoiseWeights.TensorCount}.");
            using RnnoiseWeights weights = new();
            weights.Load(tensors);
        }
        finally
        {
            foreach (Tensor tensor in tensors.Values) tensor.Dispose();
        }
    }
}
