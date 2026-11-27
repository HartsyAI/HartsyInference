using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The GPT-OSS block's self-attention through the backend's flash attention, against the loop the encoder ran
/// before: keys and values repeated to the query head count, an additive causal or sliding-window mask, and the per-head
/// sink in the softmax's denominator. That loop is kept here as the oracle.
/// <para>The tolerance was fixed before the first run at 1e-4 relative to the larger of 1 and the value. With sinks, which
/// is what the GPT-OSS checkpoint has, every backend met it on that run. Without sinks the GPU backends missed it (2.6e-4):
/// they serve a causal prefill with no sinks from their F16 kernels, as they do for every caller of this entry. Those
/// cases are held to 3e-3 on a GPU, the bound this repo's attention tests already hold F16 rows to, set after seeing that
/// miss.</para></summary>
[Collection("AttentionEvidence")]
public sealed class GptOssAttentionTests
{
    private readonly ITestOutputHelper _output;

    public GptOssAttentionTests(ITestOutputHelper output) => _output = output;

    private static Tensor Random(TensorShape shape, int seed, float scale = 1f)
    {
        Tensor tensor = new(shape, DType.F32);
        Random random = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = (random.NextSingle() * 2f - 1f) * scale;
        return tensor;
    }

    /// <summary>The encoder's former attention, verbatim in its arithmetic: F32, logits <c>q·k / √D + mask</c> with a mask
    /// of 0 or -1e30, the maximum taken over the key logits, and <c>exp(sink − max)</c> added to the denominator.</summary>
    private static float[] FormerLoop(Tensor query, Tensor key, Tensor value, Tensor? sinks, int slidingWindow)
    {
        int batch = (int)query.Shape[0], heads = (int)query.Shape[1], seqLen = (int)query.Shape[2], dim = (int)query.Shape[3];
        int group = heads / (int)key.Shape[1], kvHeads = (int)key.Shape[1];
        ReadOnlySpan<float> q = query.AsReadOnlySpan<float>(), k = key.AsReadOnlySpan<float>(), v = value.AsReadOnlySpan<float>();
        float[] output = new float[q.Length], logits = new float[seqLen];
        float scale = 1f / MathF.Sqrt(dim);
        int window = slidingWindow == 0 ? int.MaxValue : slidingWindow;
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < heads; h++)
            {
                int qHead = ((b * heads + h) * seqLen) * dim, kvHead = ((b * kvHeads + h / group) * seqLen) * dim;
                float sink = sinks is null ? 0f : sinks.AsReadOnlySpan<float>()[h];
                for (int qPos = 0; qPos < seqLen; qPos++)
                {
                    float max = float.NegativeInfinity;
                    for (int kPos = 0; kPos < seqLen; kPos++)
                    {
                        float dot = 0f;
                        for (int d = 0; d < dim; d++) dot += q[qHead + qPos * dim + d] * k[kvHead + kPos * dim + d];
                        bool visible = kPos <= qPos && (long)kPos > (long)qPos - window;
                        logits[kPos] = dot * scale + (visible ? 0f : -1e30f);
                        if (logits[kPos] > max) max = logits[kPos];
                    }
                    float sum = 0f;
                    for (int kPos = 0; kPos < seqLen; kPos++)
                    {
                        logits[kPos] = MathF.Exp(logits[kPos] - max);
                        sum += logits[kPos];
                    }
                    float denominator = sinks is null ? sum : sum + MathF.Exp(sink - max);
                    for (int kPos = 0; kPos < seqLen; kPos++)
                    {
                        float weight = logits[kPos] / denominator;
                        if (weight == 0f) continue;
                        for (int d = 0; d < dim; d++) output[qHead + qPos * dim + d] += weight * v[kvHead + kPos * dim + d];
                    }
                }
            }
        return output;
    }

    /// <summary>The shapes the Lens encoder runs (64 query heads over 8, head dim 64) at a short prompt, with and without
    /// sinks and a window, and a batch of two.</summary>
    private static void MatchesTheFormerLoop(IBackend backend, float withoutSinksTolerance = 1e-4f)
    {
        foreach ((int batch, int heads, int kvHeads, int dim, int seqLen, int window, bool withSinks) in
            ((int, int, int, int, int, int, bool)[])[(1, 64, 8, 64, 96, 0, true), (1, 64, 8, 64, 96, 40, true), (2, 64, 8, 64, 48, 16, true),
                (1, 64, 8, 64, 70, 128, true), (1, 8, 8, 64, 33, 0, false), (1, 16, 4, 64, 80, 24, false),
                // The checkpoint's own window of 128 where it binds: the 143 tokens of the real Lens prompt, and past two windows.
                (1, 64, 8, 64, 143, 128, true), (1, 64, 8, 64, 300, 128, true), (1, 64, 8, 64, 143, 0, true)])
        {
            using Tensor query = Random(new TensorShape(batch, heads, seqLen, dim), 71);
            using Tensor key = Random(new TensorShape(batch, kvHeads, seqLen, dim), 72);
            using Tensor value = Random(key.Shape, 73);
            using Tensor? sinks = withSinks ? Random(new TensorShape(heads), 74, scale: 3f) : null;
            using Tensor output = new(query.Shape, DType.F32);
            float[] expected = FormerLoop(query, key, value, sinks, window);

            GptOssEncoder.Attend(backend, output, query, key, value, sinks, window);
            backend.Sync();
            ReadOnlySpan<float> actual = output.AsReadOnlySpan<float>();
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= (withSinks ? 1e-4f : withoutSinksTolerance) * Math.Max(1f, Math.Abs(expected[i])),
                    $"batch {batch}, heads {heads}/{kvHeads}, seq {seqLen}, window {window}, sinks {withSinks}: element {i} is {actual[i]}, expected {expected[i]}");
        }
    }

    [Fact]
    public void OnTheCpuBackendItMatchesTheFormerLoop()
    {
        using CpuBackend backend = new();
        MatchesTheFormerLoop(backend);
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void OnCudaItMatchesTheFormerLoopAndNamesTheRowsThatRan()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new(0, PtxDirectory());
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(256);
        MatchesTheFormerLoop(backend, withoutSinksTolerance: 3e-3f);
        ReportRows(evidence.Snapshot());
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
            MatchesTheFormerLoop(backend, withoutSinksTolerance: 3e-3f);
            ReportRows(evidence.Snapshot());
        }
    }

    /// <summary>Which kernel rows served the calls, and whether any was the declared host reference: reported, not judged.</summary>
    private void ReportRows(AttentionEvidenceEvent[] events)
    {
        foreach (IGrouping<string, AttentionEvidenceEvent> row in events.Where(e => e.Kind == AttentionEvidenceKind.Execution)
                     .GroupBy(e => $"{e.Implementation} [{e.Branch}] heads {e.Heads}/{e.KvHeads}, batch {e.Batch}, seq {e.QueryTokens}"))
            _output.WriteLine($"{row.Key}: {row.Count()} call(s)");
        Assert.Contains(events, e => e.Kind == AttentionEvidenceKind.Execution);
        Assert.DoesNotContain(events, e => e.Kind == AttentionEvidenceKind.Repeat);
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
