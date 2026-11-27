using System.Text.Json;

namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>Reads one Duo run from a folder, for the experimental file input of the CLI route. <c>manifest.json</c> names little-endian float32
/// files for the shared Intent rows, the reference latent rows, and each block's body Intent rows and initial noise, along with the step count
/// and the flow shift. Every file's length is checked against the sample's counts before a run starts, so a bad file fails here and not in the
/// middle of a denoise. The Intent rows are supplied by the user; no Intent model runs.</summary>
public static class DuoRunFiles
{
    private const int FloatBytes = 4;

    /// <summary>Reads the run in <paramref name="directory"/> for <paramref name="sample"/>. The widths are the row widths of the text, the video and
    /// the audio rows.</summary>
    /// <exception cref="InvalidDataException">The manifest is missing, malformed, or names a file of the wrong length or with a block count that differs from the sample.</exception>
    public static DuoRun Read(string directory, DuoSampleInput sample, int textWidth, int videoWidth, int audioWidth)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        ArgumentNullException.ThrowIfNull(sample);
        if (!BitConverter.IsLittleEndian)
        {
            throw new NotSupportedException("The run files are little-endian float32; this host is big-endian.");
        }
        string manifestPath = Path.Combine(directory, "manifest.json");
        DuoRunManifest manifest = JsonSerializer.Deserialize<DuoRunManifest>(File.ReadAllText(manifestPath), new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
            ?? throw new InvalidDataException($"{manifestPath} is empty.");
        if (manifest.Steps < 1)
        {
            throw new InvalidDataException($"{manifestPath}: steps must be at least 1, got {manifest.Steps}.");
        }
        if (manifest.Blocks is null || manifest.Blocks.Count != sample.Blocks.Count)
        {
            throw new InvalidDataException($"{manifestPath} names {manifest.Blocks?.Count ?? 0} blocks; the sample has {sample.Blocks.Count}.");
        }

        float[] shared = Load(directory, manifest.SharedText, (sample.SharedDescriptionRows + sample.SharedCueRows) * textWidth, "sharedText");
        float[] reference = Load(directory, manifest.ReferenceVideo, sample.Reference.VideoRows * videoWidth, "referenceVideo");
        List<float[]> body = [], noiseVideo = [], noiseAudio = [];
        for (int block = 0; block < sample.Blocks.Count; block++)
        {
            DuoBlockFiles files = manifest.Blocks[block] ?? throw new InvalidDataException($"{manifestPath}: block {block} is null.");
            DuoBlockInput input = sample.Blocks[block];
            body.Add(Load(directory, files.BodyText, input.BodyTextRows * textWidth, $"blocks[{block}].bodyText"));
            noiseVideo.Add(Load(directory, files.NoiseVideo, input.Video.VideoRows * videoWidth, $"blocks[{block}].noiseVideo"));
            noiseAudio.Add(Load(directory, files.NoiseAudio, 2 * input.AudioSteps * audioWidth, $"blocks[{block}].noiseAudio"));
        }
        if (manifest.Shift is not { } shift || !double.IsFinite(shift) || shift <= 0)
        {
            throw new InvalidDataException($"{manifestPath}: shift must be a finite number above 0; it is {manifest.Shift?.ToString(System.Globalization.CultureInfo.InvariantCulture) ?? "missing"}.");
        }
        return new DuoRun(shared, reference, body, noiseVideo, noiseAudio, manifest.Steps, shift);
    }

    /// <summary>The path of a file named by the manifest, which must sit inside the run folder: absolute paths and any path that leaves the folder
    /// through ".." are refused.</summary>
    private static string ResolveInside(string directory, string file, string field)
    {
        if (Path.IsPathRooted(file))
        {
            throw new InvalidDataException($"{field}: {file} is an absolute path; the run folder takes relative names only.");
        }
        string root = Path.GetFullPath(directory).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
        string path = Path.GetFullPath(Path.Combine(root, file));
        if (!path.StartsWith(root, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException($"{field}: {file} resolves outside the run folder.");
        }
        return path;
    }

    private static float[] Load(string directory, string? file, int count, string field)
    {
        if (string.IsNullOrEmpty(file))
        {
            throw new InvalidDataException($"The manifest has no file for {field}.");
        }
        string path = ResolveInside(directory, file, field);
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length != count * FloatBytes)
        {
            throw new InvalidDataException($"{field}: {file} holds {bytes.Length} bytes; expected {count * FloatBytes} ({count} float32 values).");
        }
        float[] values = new float[count];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    internal sealed class DuoRunManifest
    {
        public int Steps { get; set; }

        public double? Shift { get; set; }

        public string? SharedText { get; set; }

        public string? ReferenceVideo { get; set; }

        public List<DuoBlockFiles?>? Blocks { get; set; }
    }

    internal sealed class DuoBlockFiles
    {
        public string? BodyText { get; set; }

        public string? NoiseVideo { get; set; }

        public string? NoiseAudio { get; set; }
    }
}
