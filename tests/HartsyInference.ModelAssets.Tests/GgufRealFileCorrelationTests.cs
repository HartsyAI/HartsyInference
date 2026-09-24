using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>A codec's block layout is settled against real weights, not synthetic blocks: every tensor a lower-bit
/// llama.cpp quantization of Llama-3.2-1B stores in the format under test is dequantized and correlated with the
/// same tensor from the Q8_0 file. A misread field (scale nibble order, sign table, high-bit plane) drops the
/// correlation to near zero; the floor per format is the quantization's own loss with margin.</summary>
public sealed class GgufRealFileCorrelationTests
{
    private readonly ITestOutputHelper _out;
    public GgufRealFileCorrelationTests(ITestOutputHelper o) { _out = o; }

    [Theory]
    [InlineData("IQ4_XS", "IQ4_XS", 0.98)]
    [InlineData("Q3_K_M", "Q3_K", 0.95)]
    [InlineData("Q2_K", "Q2_K", 0.90)]
    public unsafe void EveryTensorInTheFormatCorrelatesWithQ8_0(string file, string format, double floor)
    {
        string path = TestPaths.Llm.Llama32_1B(file);
        string refPath = TestPaths.Llm.Llama32_1BQ8;
        if (!File.Exists(path) || !File.Exists(refPath))
        {
            _out.WriteLine($"SKIPPED: {path} or {refPath} is not staged");
            return;
        }
        using GgufLoader quant = new();
        using GgufLoader reference = new();
        quant.Load(path);
        reference.Load(refPath);

        int examined = 0;
        double worst = 1.0;
        string worstName = "";
        foreach ((string name, GgufTensorDescriptor desc) in quant.Descriptors)
        {
            if (desc.DType.Name != format) continue;
            using Tensor q = quant.GetTensor(name);
            using Tensor r = reference.GetTensor(name);
            using Tensor qf = GgufDequantizer.Dequantize(q, DType.F32);
            using Tensor rf = GgufDequantizer.Dequantize(r, DType.F32);
            double corr = Pearson(qf.AsReadOnlySpan<float>(), rf.AsReadOnlySpan<float>());
            if (corr < worst) { worst = corr; worstName = name; }
            examined++;
        }
        string histogram = string.Join(", ", quant.Descriptors.Values.GroupBy(d => d.DType.Name).OrderBy(g => g.Key).Select(g => $"{g.Key}×{g.Count()}"));
        _out.WriteLine($"{file}: {histogram}; {examined} {format} tensors, worst correlation {worst:F5} ({worstName})");
        Assert.True(examined > 0, $"{path} stores no {format} tensor; the wrong file is staged");
        Assert.True(worst >= floor, $"{worstName}: correlation {worst:F5} with Q8_0 is below {floor}");
    }

    private static double Pearson(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        Assert.Equal(a.Length, b.Length);
        double ma = 0, mb = 0;
        for (int i = 0; i < a.Length; i++) { ma += a[i]; mb += b[i]; }
        ma /= a.Length; mb /= b.Length;
        double sab = 0, saa = 0, sbb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double da = a[i] - ma, db = b[i] - mb;
            sab += da * db; saa += da * da; sbb += db * db;
        }
        return sab / Math.Sqrt(saa * sbb);
    }
}
