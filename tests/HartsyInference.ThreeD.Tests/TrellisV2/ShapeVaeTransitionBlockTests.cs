using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

/// <summary>Transition blocks with zero conv weights: each conv output is its bias, so the skip path is checked by hand.</summary>
public sealed class ShapeVaeTransitionBlockTests
{
    [Fact]
    public void SpatialToChannel_SkipAveragesMergedChildrenAndConvAddsBias()
    {
        // Two children of one coarse cell (sub 0 and sub 7), cin 8, cout 16: the merged skip is 64 channels averaged into 16 groups of 4.
        int[] coords = { 0, 0, 0, 0, 0, 1, 1, 1 };
        using Tensor feats = Values(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 60, 70, 80 }, 2, 8);
        Dictionary<string, Tensor> w = new()
        {
            ["s2c.norm1.weight"] = Fill(1f, 8),
            ["s2c.norm1.bias"] = Zeros(8),
            ["s2c.conv1.weight"] = Zeros(2, 3, 3, 3, 8),
            ["s2c.conv1.bias"] = Zeros(2),
            ["s2c.conv2.weight"] = Zeros(16, 3, 3, 3, 16),
            ["s2c.conv2.bias"] = Fill(0.5f, 16),
        };
        using ShapeVaeSpatialToChannelBlock block = new(w, "s2c", 8, 16);
        SparseSpatialChannel.Plan plan = SparseSpatialChannel.PlanSpatialToChannel(coords, 4, 2);

        SparseTensor output = block.Forward(new CpuBackend(), ShapeVaeTestData.Sparse(coords, 4, feats), plan);

        Assert.Equal(new[] { 0, 0, 0, 0 }, output.Coords);
        Assert.Equal(2, output.Resolution);
        // Groups 0 and 1 average child 0 (1..8); groups 14 and 15 average child 7 (10..80); the rest are empty.
        float[] expected = new float[16];
        expected[0] = 2.5f + 0.5f;
        expected[1] = 6.5f + 0.5f;
        expected[14] = 25f + 0.5f;
        expected[15] = 65f + 0.5f;
        for (int i = 2; i < 14; i++) expected[i] = 0.5f;
        AssertClose(expected, ShapeVaeTestData.Read(output.Feats));
    }

    [Fact]
    public void ChannelToSpatial_ExpandsOnlyChildZeroAndRepeatsItsChannels()
    {
        // Two parents at res 2, cin 16, cout 8: the subdivision bias keeps child 0 only, and the skip repeats each
        // child-0 channel group (cin/8 = 2 channels) 4 times onto cout = 8 channels.
        int[] coords = { 0, 1, 0, 1, 0, 1, 1, 1 };
        using Tensor feats = Values(Ramp(1, 32), 2, 16);
        float[] subdivBias = { 1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f };
        Dictionary<string, Tensor> w = new()
        {
            ["c2s.norm1.weight"] = Fill(1f, 16),
            ["c2s.norm1.bias"] = Zeros(16),
            ["c2s.conv1.weight"] = Zeros(64, 3, 3, 3, 16),
            ["c2s.conv1.bias"] = Zeros(64),
            ["c2s.conv2.weight"] = Zeros(8, 3, 3, 3, 8),
            ["c2s.conv2.bias"] = Fill(0.5f, 8),
            ["c2s.to_subdiv.weight"] = Zeros(8, 16),
            ["c2s.to_subdiv.bias"] = Values(subdivBias, 8),
        };
        using ShapeVaeChannelToSpatialBlock block = new(w, "c2s", 16, 8);

        SparseTensor output = block.Forward(new CpuBackend(), ShapeVaeTestData.Sparse(coords, 2, feats));

        Assert.Equal(new[] { 0, 2, 0, 2, 0, 2, 2, 2 }, output.Coords);
        Assert.Equal(4, output.Resolution);
        // Parent 0 holds channels 1..16, parent 1 holds 17..32. Child 0 reads channels 0..1 of each parent.
        float[] expected =
        {
            1.5f, 1.5f, 1.5f, 1.5f, 2.5f, 2.5f, 2.5f, 2.5f,
            17.5f, 17.5f, 17.5f, 17.5f, 18.5f, 18.5f, 18.5f, 18.5f,
        };
        AssertClose(expected, ShapeVaeTestData.Read(output.Feats));
    }

    [Fact]
    public void ChannelToSpatial_CentreTapMainPathMatchesClosedForm()
    {
        // conv1 and conv2 are identity on the centre tap, so the output is closed form: child 0 carries
        // SiLU(LN(x))[0..7] through conv2's no-affine norm and SiLU, and the skip repeats x[0..1] four times.
        int[] coords = { 0, 1, 0, 1, 0, 1, 1, 1 };
        float[] input = Ramp(1, 32);
        using Tensor feats = Values(input, 2, 16);
        float[] subdivBias = { 1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f };
        Dictionary<string, Tensor> w = new()
        {
            ["c2s.norm1.weight"] = Fill(1f, 16),
            ["c2s.norm1.bias"] = Zeros(16),
            ["c2s.conv1.weight"] = CentreTap(64, 16),
            ["c2s.conv1.bias"] = Zeros(64),
            ["c2s.conv2.weight"] = CentreTap(8, 8),
            ["c2s.conv2.bias"] = Zeros(8),
            ["c2s.to_subdiv.weight"] = Zeros(8, 16),
            ["c2s.to_subdiv.bias"] = Values(subdivBias, 8),
        };
        using ShapeVaeChannelToSpatialBlock block = new(w, "c2s", 16, 8);

        SparseTensor output = block.Forward(new CpuBackend(), ShapeVaeTestData.Sparse(coords, 2, feats));

        float[] expected = new float[16];
        for (int parent = 0; parent < 2; parent++)
        {
            double[] row = new double[16];
            for (int c = 0; c < 16; c++) row[c] = input[parent * 16 + c];
            double[] activated = Silu(Normalize(row));
            double[] child = Silu(Normalize(activated.AsSpan(0, 8)));
            for (int c = 0; c < 8; c++)
                expected[parent * 8 + c] = (float)(child[c] + input[parent * 16 + c / 4]);
        }
        AssertClose(expected, ShapeVaeTestData.Read(output.Feats));
    }

    /// <summary>Identity on the centre tap of a 3x3x3 kernel: <c>[Cout, 3, 3, 3, Cin]</c> with <c>w[o, 1, 1, 1, o] = 1</c>.</summary>
    private static Tensor CentreTap(int cout, int cin)
    {
        Tensor t = Zeros(cout, 3, 3, 3, cin);
        Span<float> s = t.AsSpan<float>();
        for (int o = 0; o < Math.Min(cout, cin); o++) s[(((o * 3 + 1) * 3 + 1) * 3 + 1) * cin + o] = 1f;
        return t;
    }

    private static double[] Normalize(ReadOnlySpan<double> x)
    {
        double mean = 0, variance = 0;
        foreach (double v in x) mean += v;
        mean /= x.Length;
        foreach (double v in x) variance += (v - mean) * (v - mean);
        variance /= x.Length;
        double[] y = new double[x.Length];
        for (int i = 0; i < y.Length; i++) y[i] = (x[i] - mean) / Math.Sqrt(variance + 1e-6);
        return y;
    }

    private static double[] Silu(double[] x)
    {
        double[] y = new double[x.Length];
        for (int i = 0; i < y.Length; i++) y[i] = x[i] / (1 + Math.Exp(-x[i]));
        return y;
    }

    private static float[] Ramp(float start, int count)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = start + i;
        return values;
    }

    private static Tensor Values(float[] values, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static Tensor Zeros(params long[] dims) => Fill(0f, dims);

    private static Tensor Fill(float value, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        t.AsSpan<float>().Fill(value);
        return t;
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 5);
    }

    [Fact]
    public void ChannelToSpatial_NoPositiveSubdivisionLogits_ThrowsClearly()
    {
        // Every to_subdiv logit is negative, so the subdivision keeps no children and must fail with a clear message.
        int[] coords = { 0, 1, 0, 1, 0, 1, 1, 1 };
        using Tensor feats = Values(Ramp(1, 32), 2, 16);
        Dictionary<string, Tensor> w = new()
        {
            ["c2s.norm1.weight"] = Fill(1f, 16),
            ["c2s.norm1.bias"] = Zeros(16),
            ["c2s.conv1.weight"] = Zeros(64, 3, 3, 3, 16),
            ["c2s.conv1.bias"] = Zeros(64),
            ["c2s.conv2.weight"] = Zeros(8, 3, 3, 3, 8),
            ["c2s.conv2.bias"] = Fill(0.5f, 8),
            ["c2s.to_subdiv.weight"] = Zeros(8, 16),
            ["c2s.to_subdiv.bias"] = Fill(-1f, 8),
        };
        using ShapeVaeChannelToSpatialBlock block = new(w, "c2s", 16, 8);

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            block.Forward(new CpuBackend(), ShapeVaeTestData.Sparse(coords, 2, feats)));
        Assert.Contains("no children", error.Message);
    }
}
