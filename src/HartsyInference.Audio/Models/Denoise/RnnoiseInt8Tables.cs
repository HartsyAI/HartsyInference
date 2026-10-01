using System.Globalization;
using System.Text.RegularExpressions;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Audio.Models.Denoise;

/// <summary>The int8 tables RNNoise's default C build compiles in, read out of <see cref="SourceMember"/> in xiph's
/// model tarball and stored in <see cref="FileName"/>.
///
/// <para>That build runs conv2 and the three GRUs on these tables. Each has int8 weights in <see cref="Int8Tiles"/>
/// order, a float scale per output row, and a float "SU" bias that also absorbs the 127 offset of the uint8
/// activations. Each GRU's recurrent weight keeps its diagonal in float (<c>_weights_diag</c>), with zeros in its place
/// in the int8 table. conv1, dense_out and vad_dense have no int8 table and run in float, on the same values the
/// PyTorch checkpoint holds.</para>
///
/// <para>The arrays are stored under upstream's names, in upstream's order, so the file can be compared byte for
/// byte with what a C compiler makes of <c>rnnoise_data.c</c>. <see cref="RnnoiseWeights.LoadInt8Tables"/> rearranges
/// them for the model. The GRU tables use upstream's sparse format, but their index lists name every 4-column group of
/// every 8-row block, so they are dense; the conversion checks that and refuses anything else.</para>
///
/// <para>Each float literal is parsed as C parses one: as a double, then rounded to float.</para></summary>
public static class RnnoiseInt8Tables
{
    /// <summary>The file's name beside <c>rnnoise.safetensors</c>.</summary>
    public const string FileName = "rnnoise_int8.safetensors";

    /// <summary>Path, inside the tarball, of the C source holding the tables.</summary>
    public const string SourceMember = "src/rnnoise_data.c";

    private const int Conv2Outputs = RnnoiseModel.GruSize;
    private const int Conv2Inputs = RnnoiseModel.CondSize * RnnoiseModel.KernelSize;
    private const int GruOutputs = 3 * RnnoiseModel.GruSize;
    private const int GruInputs = RnnoiseModel.GruSize;

    private static readonly Regex Declaration = new(@"^static const (float|opus_int8|int) (\w+)\[(\d+)\] = \{\s*$",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    /// <summary>Every array the file holds, by upstream's name, with its element count.</summary>
    public static IReadOnlyDictionary<string, int> Arrays { get; } = BuildArrays();

    private static Dictionary<string, int> BuildArrays()
    {
        Dictionary<string, int> arrays = new(StringComparer.Ordinal)
        {
            ["conv2_weights_int8"] = Conv2Outputs * Conv2Inputs,
            ["conv2_scale"] = Conv2Outputs,
            ["conv2_subias"] = Conv2Outputs,
        };
        for (int layer = 1; layer <= 3; layer++)
        {
            foreach (string kind in new[] { "input", "recurrent" })
            {
                string prefix = $"gru{layer}_{kind}";
                arrays[$"{prefix}_weights_int8"] = GruOutputs * GruInputs;
                arrays[$"{prefix}_scale"] = GruOutputs;
                arrays[$"{prefix}_subias"] = GruOutputs;
            }
            arrays[$"gru{layer}_recurrent_weights_diag"] = GruOutputs;
        }
        return arrays;
    }

    /// <summary>Converts the tarball's <see cref="SourceMember"/> to <paramref name="outputPath"/>.</summary>
    /// <param name="metadata">Written to the output header's <c>__metadata__</c>; provenance belongs here.</param>
    public static void ConvertTarball(string tarballPath, string outputPath,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(tarballPath);
        ArgumentException.ThrowIfNullOrEmpty(outputPath);
        string directory = Path.GetDirectoryName(Path.GetFullPath(outputPath))!;
        Directory.CreateDirectory(directory);
        // Beside the output, so a failed install leaves nothing behind in a shared temp directory.
        string source = Path.Combine(directory, $".{Path.GetFileName(outputPath)}.{Guid.NewGuid():N}.c");
        try
        {
            RnnoiseCheckpoint.ExtractMember(tarballPath, SourceMember, source);
            ConvertSource(source, outputPath, metadata);
        }
        finally
        {
            File.Delete(source);
        }
    }

    /// <summary>Converts an <c>rnnoise_data.c</c> already on disk to <paramref name="outputPath"/>. An existing output
    /// is replaced only once the new file has loaded back.</summary>
    /// <param name="metadata">Written to the output header's <c>__metadata__</c>.</param>
    public static void ConvertSource(string sourcePath, string outputPath,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(sourcePath);
        ArgumentException.ThrowIfNullOrEmpty(outputPath);
        Dictionary<string, Tensor> tensors;
        using (StreamReader reader = new(sourcePath))
        {
            tensors = Parse(reader);
        }
        string staging = $"{outputPath}.{Guid.NewGuid():N}.staging";
        try
        {
            SafeTensorsWriter.Save(staging, tensors, metadata);
            Validate(staging);
            File.Move(staging, outputPath, overwrite: true);
        }
        finally
        {
            File.Delete(staging);
            foreach (Tensor tensor in tensors.Values) tensor.Dispose();
        }
    }

    /// <summary>Reads the int8 layers' arrays out of an <c>rnnoise_data.c</c>, skipping every other array, and checks
    /// that the GRU index lists describe dense matrices.</summary>
    /// <exception cref="InvalidDataException">An array is missing, has another type or length, or an index list is
    /// sparse.</exception>
    internal static Dictionary<string, Tensor> Parse(TextReader source)
    {
        Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
        Dictionary<string, int[]> indices = new(StringComparer.Ordinal);
        try
        {
            string? line;
            while ((line = source.ReadLine()) is not null)
            {
                Match declaration = Declaration.Match(line);
                if (!declaration.Success) continue;
                string type = declaration.Groups[1].Value;
                string name = declaration.Groups[2].Value;
                int count = int.Parse(declaration.Groups[3].Value, CultureInfo.InvariantCulture);
                if (Arrays.TryGetValue(name, out int expected))
                {
                    string expectedType = name.EndsWith("_weights_int8", StringComparison.Ordinal) ? "opus_int8" : "float";
                    if (type != expectedType || count != expected)
                        throw new InvalidDataException(
                            $"'{name}' is {type}[{count}], expected {expectedType}[{expected}]; not RNNoise's default model.");
                    tensors[name] = type == "float" ? ReadFloats(source, name, count) : ReadInt8(source, name, count);
                }
                else if (type == "int" && name.EndsWith("_weights_idx", StringComparison.Ordinal))
                {
                    indices[name] = ReadInts(source, name, count);
                }
                else
                {
                    SkipArray(source, name);
                }
            }

            foreach (string name in Arrays.Keys)
            {
                if (!tensors.ContainsKey(name))
                    throw new InvalidDataException($"'{name}' is missing; not RNNoise's default model source.");
            }
            foreach ((string name, int[] index) in indices)
            {
                string table = name[..^"_idx".Length] + "_int8";
                if (!Arrays.TryGetValue(table, out int weights)) continue;
                int outputs = Arrays[name[..^"_weights_idx".Length] + "_scale"];
                RequireDense(index, name, outputs, weights / outputs);
            }
            return tensors;
        }
        catch
        {
            foreach (Tensor tensor in tensors.Values) tensor.Dispose();
            throw;
        }
    }

    /// <summary>Checks that <paramref name="tables"/> holds every array of the file at its type and length.</summary>
    /// <exception cref="InvalidDataException">One is missing or differs.</exception>
    public static void Require(IReadOnlyDictionary<string, Tensor> tables)
    {
        ArgumentNullException.ThrowIfNull(tables);
        foreach ((string name, int count) in Arrays)
        {
            if (!tables.TryGetValue(name, out Tensor? tensor))
                throw new InvalidDataException($"The RNNoise int8 tables lack '{name}'. Expected {FileName} from {SourceMember}.");
            DType dtype = name.EndsWith("_weights_int8", StringComparison.Ordinal) ? DType.I8 : DType.F32;
            if (tensor.DType != dtype || tensor.ElementCount != count)
                throw new InvalidDataException(
                    $"RNNoise int8 table '{name}' is {tensor.DType} with {tensor.ElementCount} elements, expected {dtype} with {count}.");
        }
    }

    /// <summary>Upstream's sparse format lists, per 8-row block, how many 4-column groups follow and where each starts.
    /// Dense means every block lists all of them, in order.</summary>
    private static void RequireDense(int[] index, string name, int outputs, int inputs)
    {
        int position = 0;
        int groups = inputs / Int8Tiles.Cols;
        for (int block = 0; block < outputs / Int8Tiles.Rows; block++)
        {
            if (position >= index.Length || index[position] != groups)
                throw new InvalidDataException($"'{name}' block {block} is sparse; only dense int8 tables are supported.");
            position++;
            for (int group = 0; group < groups; group++, position++)
            {
                if (index[position] != group * Int8Tiles.Cols)
                    throw new InvalidDataException($"'{name}' block {block} skips or reorders columns; only dense tables are supported.");
            }
        }
        if (position != index.Length)
            throw new InvalidDataException($"'{name}' has {index.Length - position} entries past its last block.");
    }

    private static void Validate(string path)
    {
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> tensors = loader.GetAllTensors();
        try
        {
            if (tensors.Count != Arrays.Count)
                throw new InvalidDataException($"'{path}' holds {tensors.Count} tensors, expected {Arrays.Count}.");
            Require(tensors);
        }
        finally
        {
            foreach (Tensor tensor in tensors.Values) tensor.Dispose();
        }
    }

    private static Tensor ReadFloats(TextReader source, string name, int count)
    {
        Tensor tensor = new(new TensorShape(count), DType.F32);
        try
        {
            Span<float> values = tensor.AsSpan<float>();
            int filled = ReadValues(source, name, count, (token, i, span) => span[i] =
                (float)double.Parse(token, NumberStyles.Float, CultureInfo.InvariantCulture), values);
            return filled == count ? tensor : throw Short(name, filled, count);
        }
        catch
        {
            tensor.Dispose();
            throw;
        }
    }

    private static Tensor ReadInt8(TextReader source, string name, int count)
    {
        Tensor tensor = new(new TensorShape(count), DType.I8);
        try
        {
            Span<sbyte> values = tensor.AsSpan<sbyte>();
            int filled = ReadValues(source, name, count, (token, i, span) => span[i] =
                sbyte.Parse(token, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture), values);
            return filled == count ? tensor : throw Short(name, filled, count);
        }
        catch
        {
            tensor.Dispose();
            throw;
        }
    }

    private static int[] ReadInts(TextReader source, string name, int count)
    {
        int[] values = new int[count];
        int filled = ReadValues(source, name, count, (token, i, span) => span[i] =
            int.Parse(token, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture), values.AsSpan());
        return filled == count ? values : throw Short(name, filled, count);
    }

    private delegate void Store<T>(string token, int index, Span<T> values);

    /// <summary>Reads comma-separated values up to the array's closing <c>};</c> and returns how many there were.</summary>
    private static int ReadValues<T>(TextReader source, string name, int count, Store<T> store, Span<T> values)
    {
        int filled = 0;
        string? line;
        while ((line = source.ReadLine()) is not null)
        {
            if (line.TrimStart().StartsWith("};", StringComparison.Ordinal)) return filled;
            foreach (string part in line.Split(','))
            {
                string token = part.Trim();
                if (token.Length == 0) continue;
                if (filled == count)
                    throw new InvalidDataException($"'{name}' has more than its declared {count} values.");
                store(token, filled++, values);
            }
        }
        throw new InvalidDataException($"'{name}' is not closed; the source is truncated.");
    }

    private static void SkipArray(TextReader source, string name)
    {
        string? line;
        while ((line = source.ReadLine()) is not null)
        {
            if (line.TrimStart().StartsWith("};", StringComparison.Ordinal)) return;
        }
        throw new InvalidDataException($"'{name}' is not closed; the source is truncated.");
    }

    private static InvalidDataException Short(string name, int filled, int count) =>
        new($"'{name}' has {filled} values, declared {count}.");
}
