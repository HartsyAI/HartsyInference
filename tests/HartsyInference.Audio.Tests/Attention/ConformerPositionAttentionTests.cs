using HartsyInference.Audio.Models.IndexTts;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Attention;

/// <summary>The IndexTTS conformer's position-aware attention through the shared attention layer, against the host loop
/// the layer ran before: scores <c>((q+u)·k + (q+v)·p) / √d</c> with the position row taken at the key. The loop is kept
/// here, verbatim in its arithmetic, as the oracle; the layer now attends the keys <c>k + p</c> with a bias a head and
/// key.
/// <para>The tolerances were fixed before the first run: 1e-4 relative to the larger of 1 and the value on the CPU backend
/// and on Vulkan, and on CUDA the contract bound for TF32, 2e-3.</para></summary>
public sealed class ConformerPositionAttentionTests
{
    private const float Tolerance = 1e-4f, Tf32Tolerance = 2e-3f;

    private readonly ITestOutputHelper _output;

    public ConformerPositionAttentionTests(ITestOutputHelper output) => _output = output;

    private static Tensor Random(TensorShape shape, int seed, float scale = 1f)
    {
        Tensor tensor = new(shape, DType.F32);
        Random random = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = (random.NextSingle() * 2f - 1f) * scale;
        return tensor;
    }

    private static float[] FormerLoop(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v, ReadOnlySpan<float> p,
        ReadOnlySpan<float> bu, ReadOnlySpan<float> bv, int t, int heads, int d)
    {
        int c = heads * d;
        float[] output = new float[t * c], scores = new float[t];
        float invSqrtD = 1f / MathF.Sqrt(d);
        for (int head = 0; head < heads; head++)
        {
            int hOff = head * d;
            for (int i = 0; i < t; i++)
            {
                float maxS = float.NegativeInfinity;
                for (int j = 0; j < t; j++)
                {
                    float ac = 0f, bd = 0f;
                    for (int e = 0; e < d; e++)
                    {
                        ac += (q[i * c + hOff + e] + bu[hOff + e]) * k[j * c + hOff + e];
                        bd += (q[i * c + hOff + e] + bv[hOff + e]) * p[j * c + hOff + e];
                    }
                    float s = (ac + bd) * invSqrtD;
                    scores[j] = s;
                    if (s > maxS) maxS = s;
                }
                float sum = 0f;
                for (int j = 0; j < t; j++) { float e = MathF.Exp(scores[j] - maxS); scores[j] = e; sum += e; }
                float invSum = 1f / sum;
                for (int j = 0; j < t; j++)
                {
                    float a = scores[j] * invSum;
                    for (int e = 0; e < d; e++) output[i * c + hOff + e] += a * v[j * c + hOff + e];
                }
            }
        }
        return output;
    }

    /// <summary>The speaker and condition conformers (8 heads of 64) and the emotion one (4 heads of 128) at reference-clip
    /// lengths, small odd shapes, and with <paramref name="longSequence"/> 1,500 frames, which a GPU row could refuse where
    /// the loop ran anything.</summary>
    private static void MatchesTheFormerLoop(IBackend backend, float tolerance, bool longSequence)
    {
        foreach ((int t, int heads, int d) in ((int, int, int)[])[(221, 8, 64), (300, 4, 128), (1500, 8, 64), (1, 8, 64), (7, 3, 16)])
        {
            if (t > 1000 && !longSequence) continue;
            int c = heads * d;
            using Tensor k = Random(new TensorShape(1, t, c), 42), p = Random(new TensorShape(1, t, c), 43);
            using Tensor bu = Random(new TensorShape(heads, d), 44, 0.5f), bv = Random(new TensorShape(heads, d), 45, 0.5f);
            using Tensor queries = Random(new TensorShape(1, t, c), 41), values = Random(new TensorShape(1, t, c), 46);
            float[] expected = FormerLoop(queries.AsReadOnlySpan<float>(), k.AsReadOnlySpan<float>(), values.AsReadOnlySpan<float>(),
                p.AsReadOnlySpan<float>(), bu.AsReadOnlySpan<float>(), bv.AsReadOnlySpan<float>(), t, heads, d);

            // The site takes its query and value over, so it gets copies.
            Tensor q = new(queries.Shape, DType.F32), v = new(values.Shape, DType.F32);
            queries.AsReadOnlySpan<float>().CopyTo(q.AsSpan<float>());
            values.AsReadOnlySpan<float>().CopyTo(v.AsSpan<float>());
            using Tensor output = IndexTtsConformerLayer.Attend(backend, q, k, v, p, bu, bv, heads, d);
            Assert.Equal(new TensorShape(1, t, c), output.Shape);
            ReadOnlySpan<float> actual = output.AsReadOnlySpan<float>();
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1f, Math.Abs(expected[i])),
                    $"{t} frames, {heads} heads of {d}: element {i} is {actual[i]}, expected {expected[i]}");
        }
    }

    [Fact]
    public void OnTheCpuBackendItMatchesTheFormerLoop()
    {
        using CpuBackend backend = new();
        MatchesTheFormerLoop(backend, Tolerance, longSequence: false);
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void OnCudaItMatchesTheFormerLoopAndNamesTheRowsThatRan()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new(0, PtxDirectory());
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(256);
        try { MatchesTheFormerLoop(backend, Tf32Tolerance, longSequence: true); }
        finally { ReportRows(evidence.Snapshot()); }
        Assert.Equal(5, evidence.Snapshot().Count(e => e.Kind == AttentionEvidenceKind.Execution));
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void OnVulkanItMatchesTheFormerLoopAndNamesTheRowsThatRan()
    {
        VulkanBackend backend;
        try { backend = new VulkanBackend(); }
        catch (Exception ex) { _output.WriteLine($"SKIPPED: Vulkan unavailable ({ex.Message})"); return; }
        using (backend)
        {
            using AttentionEvidenceSession evidence = AttentionEvidence.Begin(256);
            try { MatchesTheFormerLoop(backend, Tolerance, longSequence: true); }
            finally { ReportRows(evidence.Snapshot()); }
            Assert.Equal(5, evidence.Snapshot().Count(e => e.Kind == AttentionEvidenceKind.Execution));
        }
    }

    /// <summary>Which kernel rows served the calls, and what else was recorded: reported also when a comparison fails.</summary>
    private void ReportRows(AttentionEvidenceEvent[] events)
    {
        foreach (IGrouping<string, AttentionEvidenceEvent> row in events.Where(e => e.Kind == AttentionEvidenceKind.Execution)
                     .GroupBy(e => $"{e.Implementation} [{e.Branch}] heads {e.Heads}, seq {e.QueryTokens}"))
            _output.WriteLine($"{row.Key}: {row.Count()} call(s)");
        foreach (IGrouping<AttentionEvidenceKind, AttentionEvidenceEvent> kind in events.Where(e => e.Kind != AttentionEvidenceKind.Execution).GroupBy(e => e.Kind))
            _output.WriteLine($"{kind.Key}: {kind.Count()} record(s)");
    }

    private static string PtxDirectory()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        throw new DirectoryNotFoundException("HartsyInference.Cuda/Ptx was not found above the test binary.");
    }
}
