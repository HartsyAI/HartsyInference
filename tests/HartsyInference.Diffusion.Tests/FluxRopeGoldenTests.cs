using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Bit-exact regression for FluxRope host tables and the Core builder, against tables captured before the Core
/// RoPE migration (<c>Fixtures/Rope/flux_host_tables.golden.txt</c>).</summary>
public sealed unsafe class FluxRopeGoldenTests
{
    [Theory]
    [InlineData("flux_real_grid")]
    [InlineData("flux_kontext_reference")]
    [InlineData("flux_fractional_negative")]
    [InlineData("four_axis_theta256")]
    public void FluxRope_HostTables_MatchGoldenBitForBit(string caseName)
    {
        GoldenCase golden = GoldenCase.Load(caseName);
        Tensor posIds = PositionTensor(golden.Positions, golden.Tokens, golden.NumAxes);
        FluxRope rope = new(golden.Axes, golden.Theta);
        rope.Precompute(posIds);
        posIds.Dispose();

        int[] actualCos = new int[golden.Cos.Length];
        int[] actualSin = new int[golden.Sin.Length];
        ExtractFluxRopeBits(rope, golden.Tokens, golden.Pairs, actualCos, actualSin);

        Assert.Equal(golden.Cos, actualCos);
        Assert.Equal(golden.Sin, actualSin);
    }

    [Theory]
    [InlineData("flux_real_grid")]
    [InlineData("flux_kontext_reference")]
    [InlineData("flux_fractional_negative")]
    [InlineData("four_axis_theta256")]
    public void CoreBuilder_MatchesGoldenBitForBit(string caseName)
    {
        GoldenCase golden = GoldenCase.Load(caseName);
        RopeFrequencyPlan plan = RopeFrequencyPlan.Create(new RopeSpec { AxisDims = golden.Axes, Theta = golden.Theta });
        float[] cos = new float[golden.Cos.Length];
        float[] sin = new float[golden.Sin.Length];

        MultiAxisRopeTables.ComputePairs(plan, golden.Positions, golden.Tokens, RopeAngleOrder.MultiplyByInverse, cos, sin);

        for (int i = 0; i < cos.Length; i++)
        {
            Assert.True(SameTableEntry(golden.Cos[i], BitConverter.SingleToInt32Bits(cos[i])), $"cos[{i}]");
            Assert.True(SameTableEntry(golden.Sin[i], BitConverter.SingleToInt32Bits(sin[i])), $"sin[{i}]");
        }
    }

    /// <summary>The fixture was read back through a rotation of a unit vector, which turns a negative-zero sine into +0.
    /// Every other entry must match bit for bit. Negative-zero entries are compared bitwise against the legacy
    /// implementation in the migration check, not against this fixture.</summary>
    private static bool SameTableEntry(int golden, int actual) =>
        golden == actual || ((golden & 0x7FFFFFFF) == 0 && (actual & 0x7FFFFFFF) == 0);

    [Fact]
    public void Precompute_RejectsPositionsWithAnotherAxisCount()
    {
        FluxRope rope = new([16, 56, 56], 10000);
        Tensor twoAxisIds = PositionTensor([0f, 1f, 2f, 3f], 2, 2);

        Assert.ThrowsAny<ArgumentException>(() => rope.Precompute(twoAxisIds));
        twoAxisIds.Dispose();
    }

    private static Tensor PositionTensor(float[] positions, int tokens, int numAxes)
    {
        Tensor tensor = new(new TensorShape(tokens, numAxes), DType.F32);
        float* ptr = (float*)tensor.DataPointer;
        for (int i = 0; i < positions.Length; i++) ptr[i] = positions[i];
        return tensor;
    }

    /// <summary>Reads FluxRope's host cos and sin by rotating a unit vector: <c>(1, 0)</c> becomes <c>(cos, sin)</c> per pair.</summary>
    private static void ExtractFluxRopeBits(FluxRope rope, int tokens, int pairs, int[] cosBits, int[] sinBits)
    {
        int headDim = pairs * 2;
        Tensor q = new(new TensorShape(1, 1, tokens, headDim), DType.F32);
        float* qp = (float*)q.DataPointer;
        for (int s = 0; s < tokens; s++)
        {
            for (int i = 0; i < pairs; i++)
            {
                qp[s * headDim + 2 * i] = 1f;
                qp[s * headDim + 2 * i + 1] = 0f;
            }
        }
        rope.ForwardSingle(q, 1, 1, tokens);

        for (int s = 0; s < tokens; s++)
        {
            for (int i = 0; i < pairs; i++)
            {
                cosBits[s * pairs + i] = BitConverter.SingleToInt32Bits(qp[s * headDim + 2 * i]);
                sinBits[s * pairs + i] = BitConverter.SingleToInt32Bits(qp[s * headDim + 2 * i + 1]);
            }
        }
        q.Dispose();
    }

    private sealed record GoldenCase(
        int[] Axes,
        int Theta,
        int Tokens,
        int NumAxes,
        int Pairs,
        float[] Positions,
        int[] Cos,
        int[] Sin)
    {
        public static GoldenCase Load(string name)
        {
            string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Rope", "flux_host_tables.golden.txt");
            string[] lines = File.ReadAllLines(path);
            int start = Array.FindIndex(lines, line => line.StartsWith($"case {name} ", StringComparison.Ordinal));
            if (start < 0)
                throw new InvalidOperationException($"Golden case '{name}' is missing from {path}.");

            string[] header = lines[start].Split(' ');
            int[] axes = header[3].Split(',').Select(int.Parse).ToArray();
            int theta = int.Parse(header[5], System.Globalization.CultureInfo.InvariantCulture);
            int tokens = int.Parse(header[7], System.Globalization.CultureInfo.InvariantCulture);
            int pairs = int.Parse(header[9], System.Globalization.CultureInfo.InvariantCulture);
            int numAxes = int.Parse(header[11], System.Globalization.CultureInfo.InvariantCulture);

            float[] positions = new float[tokens * numAxes];
            int[] cos = new int[tokens * pairs];
            int[] sin = new int[tokens * pairs];
            for (int s = 0; s < tokens; s++)
            {
                string[] pos = lines[start + 1 + s].Split(' ');
                for (int a = 0; a < numAxes; a++) positions[s * numAxes + a] = BitConverter.Int32BitsToSingle(ParseHex(pos[1 + a]));
                string[] cosRow = lines[start + 1 + tokens + s].Split(' ');
                string[] sinRow = lines[start + 1 + 2 * tokens + s].Split(' ');
                for (int i = 0; i < pairs; i++)
                {
                    cos[s * pairs + i] = ParseHex(cosRow[1 + i]);
                    sin[s * pairs + i] = ParseHex(sinRow[1 + i]);
                }
            }
            return new GoldenCase(axes, theta, tokens, numAxes, pairs, positions, cos, sin);
        }

        private static int ParseHex(string hex) => int.Parse(hex, System.Globalization.NumberStyles.HexNumber, System.Globalization.CultureInfo.InvariantCulture);
    }
}
