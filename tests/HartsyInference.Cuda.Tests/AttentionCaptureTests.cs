using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>diagnostics.attentionCaptureDir: every call at least 1024 by 1024 is listed per shape, a listed ordinal writes the
/// chosen heads' rows exactly as the call saw them, and nothing is read back while the stream captures.</summary>
[Collection("CudaSerial")]
public sealed class AttentionCaptureTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData("3", "F16-b1-h24x24-q4096-k4096-d128-HeadMajor", 3, true)]
    [InlineData("3", "F16-b1-h24x24-q4096-k4096-d128-HeadMajor", 2, false)]
    [InlineData("q4096@2", "F16-b1-h24x24-q4096-k4096-d128-HeadMajor", 2, true)]
    [InlineData("q4096@2", "F16-b1-h24x24-q8192-k8192-d128-HeadMajor", 2, false)]
    [InlineData(" 7 , q4096@2 ", "F16-b1-h24x24-q4096-k4096-d128-HeadMajor", 7, true)]
    [InlineData("x,@,q4096@,q4096@two,,", "F16-b1-h24x24-q4096-k4096-d128-HeadMajor", 0, false)]
    [InlineData("@5", "F16-b1-h24x24-q4096-k4096-d128-HeadMajor", 5, true)]
    [InlineData(null, "F16-b1-h24x24-q4096-k4096-d128-HeadMajor", 0, false)]
    public void CaptureWanted_ReadsOrdinalsAndShapeScopedOrdinals(string? list, string shape, int ordinal, bool wanted) =>
        Assert.Equal(wanted, CudaBackend.CaptureWanted(list, shape, ordinal));

    [Theory]
    [InlineData(1, 4, new[] { 0 })]
    [InlineData(3, 4, new[] { 0, 1, 2 })]
    [InlineData(24, 4, new[] { 0, 7, 15, 23 })]
    [InlineData(6, 4, new[] { 0, 1, 3, 5 })]
    [InlineData(1100, 2, new[] { 0, 1099 })]
    public void Spread_KeepsTheEndsAndSpacesTheRest(int extent, int count, int[] expected) =>
        Assert.Equal(expected, CudaBackend.Spread(extent, count));

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void ListedCalls_WriteTheOperandsRowsAndNothingUnderCapture()
    {
        if (!BackendGate.TryOpen("cuda", output.WriteLine, out IBackend? backend)) return;
        using CudaBackend cuda = (CudaBackend)backend!;
        string dir = Path.Combine(Path.GetTempPath(), $"attn-capture-{Guid.NewGuid():N}");
        const int Heads = 6, Tokens = 1100, D = 64;
        TensorShape shape = new(1, Heads, Tokens, D);
        float[] values = new float[shape.ElementCount];
        Random rng = new(3);
        for (int i = 0; i < values.Length; i++) values[i] = (float)rng.NextDouble();
        using Tensor qh = new(shape, DType.F32), q = new(shape, DType.F32), o1 = new(shape, DType.F32), o2 = new(shape, DType.F32);
        values.CopyTo(qh.AsSpan<float>());
        cuda.Scale(q, qh, 1f);
        KnobStore.Set(EngineKnobs.AttentionCaptureDir, dir);
        KnobStore.Set(EngineKnobs.AttentionCaptureCalls, "q999@0,q1100@1");
        try
        {
            cuda.ScaledDotProductAttention(o1, q, q, q, null, 0.125f);
            cuda.ScaledDotProductAttention(o2, q, q, q, null, 0.125f);
            cuda.StepGraphBegin();
            using (Tensor o3 = new(shape, DType.F32)) cuda.ScaledDotProductAttention(o3, q, q, q, null, 0.125f);
            cuda.StepGraphEndAndLaunch();
            cuda.StepGraphReset();
            string[] calls = File.ReadAllLines(Path.Combine(dir, "calls.tsv"));
            Assert.Equal(2, calls.Length);
            string[] captured = Directory.GetDirectories(dir);
            string only = Assert.Single(captured);
            Assert.EndsWith("-call1", only);
            Assert.Contains("\"capturedHeads\":[0,1,3,5]", File.ReadAllText(Path.Combine(only, "meta.json")));
            byte[] k = File.ReadAllBytes(Path.Combine(only, "k.bin"));
            Assert.Equal(4L * Tokens * D * 4, k.Length);
            int[] heads = [0, 1, 3, 5];
            for (int i = 0; i < heads.Length; i++)
                for (int t = 0; t < Tokens; t += 97)
                    for (int c = 0; c < D; c++)
                        Assert.Equal(values[((long)heads[i] * Tokens + t) * D + c], BitConverter.ToSingle(k, (int)((((long)i * Tokens) + t) * D + c) * 4));
            Assert.Equal(4L * 512 * D * 4, new FileInfo(Path.Combine(only, "q.bin")).Length);
            // A directory that cannot be made is logged; the call itself runs.
            string blocker = Path.Combine(dir, "not-a-directory");
            File.WriteAllText(blocker, "");
            KnobStore.Set(EngineKnobs.AttentionCaptureDir, blocker);
            using Tensor o4 = new(shape, DType.F32);
            cuda.ScaledDotProductAttention(o4, q, q, q, null, 0.125f);
            Assert.Equal(o1.AsReadOnlySpan<float>().ToArray(), o4.AsReadOnlySpan<float>().ToArray());
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.AttentionCaptureDir);
            KnobStore.Clear(EngineKnobs.AttentionCaptureCalls);
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }
}
