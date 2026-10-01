using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>How <see cref="RnnoiseWeights.LoadInt8Tables"/> turns upstream's arrangement into the model's. Each table
/// is filled with values that name their own position, so a misplaced element shows up as the wrong name.</summary>
public sealed class RnnoiseInt8WeightsTests
{
    private const int Gru = RnnoiseModel.GruSize;

    /// <summary>Upstream's GRU rows run (z, r, n); the model's run (r, z, n). Every row of the table, and its scale,
    /// SU bias and diagonal, must land in its gate's new block.</summary>
    [Fact]
    public void GruTables_ComeOutInPyTorchsGateOrder()
    {
        using RnnoiseWeights weights = LoadedWithLabelledTables();
        for (int layer = 0; layer < 3; layer++)
        {
            sbyte[] input = Logical(weights.GruInputInt8[layer]!, 3 * Gru, Gru);
            float[] diag = weights.GruRecurrentDiag[layer]!.AsSpan<float>().ToArray();
            float[] scale = weights.GruInputScale[layer]!.AsSpan<float>().ToArray();
            int[] upstreamGate = [1, 0, 2];   // model gate r is upstream's second block, z its first, n its third
            for (int gate = 0; gate < 3; gate++)
            {
                foreach (int i in new[] { 0, 7, 8, 383 })
                {
                    int row = gate * Gru + i;
                    int upstreamRow = upstreamGate[gate] * Gru + i;
                    Assert.Equal(GruLabel(upstreamRow, 5), input[row * Gru + 5]);
                    Assert.Equal(upstreamRow, scale[row]);
                    Assert.Equal(-upstreamRow, diag[row]);
                }
            }
        }
    }

    /// <summary>Upstream's conv2 reads its window time-major, <c>t·128 + c</c>; the model's window is channel-major,
    /// <c>c·3 + t</c>. Each weight must follow its (channel, tap).</summary>
    [Fact]
    public void Conv2Table_ReadsTheModelsChannelMajorWindow()
    {
        using RnnoiseWeights weights = LoadedWithLabelledTables();
        const int channels = RnnoiseModel.CondSize, taps = RnnoiseModel.KernelSize, inputs = channels * taps;
        sbyte[] model = Logical(weights.Conv2Int8!, Gru, inputs);
        foreach (int row in new[] { 0, 9, 383 })
        {
            for (int c = 0; c < channels; c += 37)
            {
                for (int t = 0; t < taps; t++)
                    Assert.Equal(Conv2Label(row, t * channels + c), model[row * inputs + c * taps + t]);
            }
        }
    }

    [Fact]
    public void Int8Tables_NeedTheFloatWeightsFirst()
    {
        using RnnoiseWeights weights = new();
        Dictionary<string, Tensor> tables = LabelledTables();
        try
        {
            Assert.Throws<InvalidOperationException>(() => weights.LoadInt8Tables(tables));
            Assert.Equal(RnnoisePrecision.Float, weights.Precision);
        }
        finally
        {
            foreach (Tensor tensor in tables.Values) tensor.Dispose();
        }
    }

    /// <summary>The selection point: F32 unless int8 is asked for, even with the tables sitting beside the weights;
    /// int8 finds them there by default.</summary>
    [Fact]
    public void LoadFile_IsFloatUnlessAskedForInt8()
    {
        string directory = Directory.CreateTempSubdirectory("rnnoise-precision-").FullName;
        try
        {
            string weightsPath = Path.Combine(directory, "rnnoise.safetensors");
            Save(weightsPath, FloatWeights());
            Save(Path.Combine(directory, RnnoiseInt8Tables.FileName), LabelledTables());

            using (RnnoiseWeights f32 = RnnoiseWeights.LoadFile(weightsPath))
            {
                Assert.Equal(RnnoisePrecision.Float, f32.Precision);
                Assert.Null(f32.Conv2Int8);
            }
            using RnnoiseWeights int8 = RnnoiseWeights.LoadFile(weightsPath, RnnoisePrecision.Int8);
            Assert.Equal(RnnoisePrecision.Int8, int8.Precision);
            Assert.NotNull(int8.Conv2Int8);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static RnnoiseWeights LoadedWithLabelledTables()
    {
        RnnoiseWeights weights = new();
        VoiceFrontendAllocationTests.Load(VoiceFrontendAllocationTests.RnnoiseLayout, seed: 1, weights.Load);
        Dictionary<string, Tensor> tables = LabelledTables();
        try
        {
            weights.LoadInt8Tables(tables);
        }
        finally
        {
            foreach (Tensor tensor in tables.Values) tensor.Dispose();
        }
        return weights;
    }

    /// <summary>Upstream-arranged tables whose every value encodes where it sits.</summary>
    private static Dictionary<string, Tensor> LabelledTables()
    {
        Dictionary<string, Tensor> tables = new(StringComparer.Ordinal);
        foreach ((string name, int count) in RnnoiseInt8Tables.Arrays)
        {
            if (name.EndsWith("_weights_int8", StringComparison.Ordinal))
            {
                bool conv = name.StartsWith("conv2", StringComparison.Ordinal);
                int outputs = conv ? Gru : 3 * Gru, inputs = count / outputs;
                sbyte[] logical = new sbyte[count];
                for (int row = 0; row < outputs; row++)
                {
                    for (int col = 0; col < inputs; col++)
                        logical[row * inputs + col] = conv ? Conv2Label(row, col) : GruLabel(row, col);
                }
                Tensor tiles = new(new TensorShape(count), DType.I8);
                Int8Tiles.Pack(logical, outputs, inputs, tiles.AsSpan<sbyte>());
                tables[name] = tiles;
                continue;
            }
            Tensor vector = new(new TensorShape(count), DType.F32);
            Span<float> values = vector.AsSpan<float>();
            bool diag = name.EndsWith("_diag", StringComparison.Ordinal);
            for (int i = 0; i < count; i++) values[i] = diag ? -i : i;
            tables[name] = vector;
        }
        return tables;
    }

    private static sbyte GruLabel(int row, int col) => (sbyte)((row * 7 + col) % 251 - 125);

    private static sbyte Conv2Label(int row, int col) => (sbyte)((row * 13 + col * 5) % 253 - 126);

    private static sbyte[] Logical(Tensor tiles, int outputs, int inputs)
    {
        sbyte[] logical = new sbyte[outputs * inputs];
        Int8Tiles.Unpack(tiles.AsSpan<sbyte>(), outputs, inputs, logical);
        return logical;
    }

    private static Dictionary<string, Tensor> FloatWeights()
    {
        Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
        Random rng = new(3);
        foreach ((string name, long[] shape) in VoiceFrontendAllocationTests.RnnoiseLayout)
        {
            Tensor tensor = new(new TensorShape(shape), DType.F32);
            Span<float> values = tensor.AsSpan<float>();
            for (int i = 0; i < values.Length; i++) values[i] = (float)((rng.NextDouble() * 2 - 1) * 0.05);
            tensors[name] = tensor;
        }
        return tensors;
    }

    private static void Save(string path, Dictionary<string, Tensor> tensors)
    {
        try
        {
            SafeTensorsWriter.Save(path, tensors);
        }
        finally
        {
            foreach (Tensor tensor in tensors.Values) tensor.Dispose();
        }
    }
}
