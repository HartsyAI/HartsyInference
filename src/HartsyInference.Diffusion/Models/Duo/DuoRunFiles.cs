using System.Text.Json;

namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>Reads one Duo run from a folder, for the experimental file input of the CLI route. <c>manifest.json</c> names little-endian float32
/// files for the shared Intent rows, the reference latent rows, each block's body Intent rows and initial noise, and for the distilled
/// solver each block's re-noise (<c>renoiseVideo</c> and <c>renoiseAudio</c>), along with the step count and the flow shift. Every file's length is checked against the sample's counts before a run starts, so a bad file fails here and not in the
/// middle of a denoise. The Intent rows are supplied by the user; no Intent model runs.</summary>
public static class DuoRunFiles
{
    private const int FloatBytes = 4;

    private static readonly JsonSerializerOptions _manifestJson = new() { PropertyNameCaseInsensitive = true };

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
        DuoRunManifest manifest = ReadManifest(manifestPath);
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
        List<DuoRenoise> renoise = [];
        for (int block = 0; block < sample.Blocks.Count; block++)
        {
            DuoBlockFiles files = manifest.Blocks[block] ?? throw new InvalidDataException($"{manifestPath}: block {block} is null.");
            DuoBlockInput input = sample.Blocks[block];
            body.Add(Load(directory, files.BodyText, input.BodyTextRows * textWidth, $"blocks[{block}].bodyText"));
            noiseVideo.Add(Load(directory, files.NoiseVideo, input.Video.VideoRows * videoWidth, $"blocks[{block}].noiseVideo"));
            noiseAudio.Add(Load(directory, files.NoiseAudio, 2 * input.AudioSteps * audioWidth, $"blocks[{block}].noiseAudio"));
            if (files.RenoiseVideo is not null || files.RenoiseAudio is not null)
            {
                // One draw per step but the last, step-major, each laid out like the initial noise.
                int draws = manifest.Steps - 1;
                renoise.Add(new DuoRenoise(
                    Load(directory, files.RenoiseVideo, checked(draws * noiseVideo[^1].Length), $"blocks[{block}].renoiseVideo"),
                    Load(directory, files.RenoiseAudio, checked(draws * noiseAudio[^1].Length), $"blocks[{block}].renoiseAudio")));
            }
        }
        if (manifest.Shift is not { } shift || !double.IsFinite(shift) || shift <= 0)
        {
            throw new InvalidDataException($"{manifestPath}: shift must be a finite number above 0; it is {manifest.Shift?.ToString(System.Globalization.CultureInfo.InvariantCulture) ?? "missing"}.");
        }
        if (renoise.Count != 0 && renoise.Count != sample.Blocks.Count)
        {
            throw new InvalidDataException($"{manifestPath}: {renoise.Count} of {sample.Blocks.Count} blocks name re-noise; give all or none.");
        }
        return new DuoRun(shared, reference, body, noiseVideo, noiseAudio, manifest.Steps, shift)
        {
            Renoise = renoise.Count == 0 ? null : renoise,
        };
    }

    /// <summary>The Intent row counts of the run in <paramref name="directory"/>, from the file lengths, for a caller that builds the sample
    /// from them. File names follow the rule <see cref="Read"/> applies.</summary>
    /// <exception cref="InvalidDataException">The manifest is missing or malformed, or a text file is missing, outside the folder or not whole
    /// rows.</exception>
    internal static DuoRunShape ReadShape(string directory, int textWidth)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(textWidth);
        string manifestPath = Path.Combine(directory, "manifest.json");
        DuoRunManifest manifest = ReadManifest(manifestPath);
        if (manifest.Blocks is null)
        {
            throw new InvalidDataException($"{manifestPath} names no blocks.");
        }
        int shared = Rows(directory, manifest.SharedText, textWidth, "sharedText");
        int[] body = new int[manifest.Blocks.Count];
        for (int block = 0; block < body.Length; block++)
        {
            DuoBlockFiles files = manifest.Blocks[block] ?? throw new InvalidDataException($"{manifestPath}: block {block} is null.");
            body[block] = Rows(directory, files.BodyText, textWidth, $"blocks[{block}].bodyText");
        }
        return new DuoRunShape(shared, body);
    }

    private static DuoRunManifest ReadManifest(string path)
    {
        if (!File.Exists(path))
        {
            throw new InvalidDataException($"{path} does not exist.");
        }
        try
        {
            return JsonSerializer.Deserialize<DuoRunManifest>(File.ReadAllText(path), _manifestJson)
                ?? throw new InvalidDataException($"{path} is empty.");
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"{path} is not a run manifest: {ex.Message}", ex);
        }
    }

    private static int Rows(string directory, string? file, int width, string field)
    {
        if (string.IsNullOrEmpty(file))
        {
            throw new InvalidDataException($"The manifest has no file for {field}.");
        }
        FileInfo info = new(ResolveInside(directory, file, field));
        long rowBytes = (long)width * FloatBytes;
        if (!info.Exists || info.Length % rowBytes != 0)
        {
            throw new InvalidDataException($"{field}: {file} must exist and hold whole rows of {width} float32 values.");
        }
        return checked((int)(info.Length / rowBytes));
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

        public string? RenoiseVideo { get; set; }

        public string? RenoiseAudio { get; set; }
    }
}
