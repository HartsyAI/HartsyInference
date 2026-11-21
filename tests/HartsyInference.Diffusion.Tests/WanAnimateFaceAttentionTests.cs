using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The Wan-Animate face adapter's frame-local attention through the shared attention layer, against the host loop
/// the block ran before: each frame's group of latent tokens attends that frame's motion tokens only. The loop is kept
/// here, verbatim in its arithmetic, as the oracle.
/// <para>The tolerances were fixed before the first run: 1e-4 relative to the larger of 1 and the value on the CPU backend
/// and on Vulkan, and on CUDA the contract bound for TF32, 2e-3, because CUDA multiplies F32 attention in TF32 for every
/// caller of the shared layer.</para></summary>
public sealed class WanAnimateFaceAttentionTests
{
    private const float Tolerance = 1e-4f, Tf32Tolerance = 2e-3f;

    private readonly ITestOutputHelper _output;

    public WanAnimateFaceAttentionTests(ITestOutputHelper output) => _output = output;

    private static Tensor Random(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        Random random = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = random.NextSingle() * 2f - 1f;
        return tensor;
    }

    /// <summary>The block's former loop: <c>q [S, dim]</c> in <paramref name="frames"/> groups over <c>k, v [frames·n, dim]</c>.</summary>
    private static float[] FormerLoop(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v, int frames, int groupLen, int n, int heads,
        int headDim)
    {
        int dim = heads * headDim;
        float[] output = new float[frames * groupLen * dim], scores = new float[n];
        float scale = 1f / MathF.Sqrt(headDim);
        for (int ti = 0; ti < frames; ti++)
            for (int gi = 0; gi < groupLen; gi++)
            {
                int qRow = ti * groupLen + gi;
                for (int h = 0; h < heads; h++)
                {
                    int qOff = qRow * dim + h * headDim;
                    float maxS = float.NegativeInfinity;
                    for (int j = 0; j < n; j++)
                    {
                        int kOff = (ti * n + j) * dim + h * headDim;
                        float dot = 0;
                        for (int d = 0; d < headDim; d++) dot += q[qOff + d] * k[kOff + d];
                        scores[j] = dot * scale;
                        if (scores[j] > maxS) maxS = scores[j];
                    }
                    float sum = 0;
                    for (int j = 0; j < n; j++) { scores[j] = MathF.Exp(scores[j] - maxS); sum += scores[j]; }
                    float invSum = 1f / sum;
                    for (int d = 0; d < headDim; d++)
                    {
                        float acc = 0;
                        for (int j = 0; j < n; j++) acc += scores[j] * v[((ti * n + j) * dim) + h * headDim + d];
                        output[qOff + d] = acc * invSum;
                    }
                }
            }
        return output;
    }

    /// <summary>The 14B model's 40 heads of 128 at a reduced token count, small odd shapes, and with
    /// <paramref name="fullSize"/> the token count of a 21-frame 480p clip, which a GPU row could refuse where the loop ran
    /// anything. The loop takes seconds at that size, so the CPU lane leaves it out.</summary>
    private static void MatchesTheFormerLoop(IBackend backend, float tolerance, bool fullSize)
    {
        foreach ((int frames, int groupLen, int n, int heads, int headDim) in ((int, int, int, int, int)[])[(5, 390, 5, 40, 128),
            (21, 1560, 5, 40, 128), (1, 7, 3, 2, 16), (3, 4, 1, 4, 8), (4, 9, 6, 3, 32)])
        {
            if (frames * groupLen > 10_000 && !fullSize) continue;
            int dim = heads * headDim;
            using Tensor q = Random(new TensorShape(frames * groupLen, dim), 81);
            using Tensor k = Random(new TensorShape(frames * n, dim), 82);
            using Tensor v = Random(new TensorShape(frames * n, dim), 83);
            float[] expected = FormerLoop(q.AsReadOnlySpan<float>(), k.AsReadOnlySpan<float>(), v.AsReadOnlySpan<float>(), frames, groupLen, n, heads,
                headDim);

            using Tensor output = WanAnimateFaceBlock.Attend(backend, q, k, v, frames, heads, headDim);
            Assert.Equal(q.Shape, output.Shape);
            ReadOnlySpan<float> actual = output.AsReadOnlySpan<float>();
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1f, Math.Abs(expected[i])),
                    $"{frames} frames of {groupLen} over {n}, {heads} heads of {headDim}: element {i} is {actual[i]}, expected {expected[i]}");
        }
    }

    [Fact]
    public void OnTheCpuBackendItMatchesTheFormerLoop()
    {
        using CpuBackend backend = new();
        MatchesTheFormerLoop(backend, Tolerance, fullSize: false);
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void OnCudaItMatchesTheFormerLoopAndNamesTheRowsThatRan()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new(0, PtxDirectory());
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(256);
        try { MatchesTheFormerLoop(backend, Tf32Tolerance, fullSize: true); }
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
            try { MatchesTheFormerLoop(backend, Tolerance, fullSize: true); }
            finally { ReportRows(evidence.Snapshot()); }
            Assert.Equal(5, evidence.Snapshot().Count(e => e.Kind == AttentionEvidenceKind.Execution));
        }
    }

    /// <summary>Which kernel rows served the calls: reported also when a comparison fails.</summary>
    private void ReportRows(AttentionEvidenceEvent[] events)
    {
        foreach (IGrouping<string, AttentionEvidenceEvent> row in events.Where(e => e.Kind == AttentionEvidenceKind.Execution)
                     .GroupBy(e => $"{e.Implementation} [{e.Branch}] batch {e.Batch}, heads {e.Heads}, seq {e.QueryTokens}"))
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
