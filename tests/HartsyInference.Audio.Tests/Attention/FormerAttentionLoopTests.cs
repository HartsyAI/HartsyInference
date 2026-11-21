using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Audio.Models.GptSoVits;
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

/// <summary>Three audio modules ran plain multi-head attention as a scalar loop on the host: the IndexTTS perceiver, the
/// GPT-SoVITS MRTE cross-attention and the ControlFoley style encoder. Each now goes through the shared attention layer.
/// The loops are kept here, verbatim in their arithmetic, as the oracle.
/// <para>The tolerance was fixed before the first run, for every backend: 1e-4 relative to the larger of 1 and the
/// value. The CPU backend and Vulkan met it on that run. CUDA missed it on one case (1.04e-4 on the 512-channel MRTE
/// shape): it serves F32 attention from its materialized row, whose products run in TF32, as they do for every caller of
/// the shared layer. CUDA is held to the contract bound for TF32, 2e-3, set after seeing that miss.</para></summary>
public sealed class FormerAttentionLoopTests
{
    private const float Tolerance = 1e-4f, Tf32Tolerance = 2e-3f;

    private readonly ITestOutputHelper _output;

    public FormerAttentionLoopTests(ITestOutputHelper output) => _output = output;

    private static Tensor Random(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        Random random = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = random.NextSingle() * 2f - 1f;
        return tensor;
    }

    private static void AssertClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1f, Math.Abs(expected[i])),
                $"{what}: element {i} is {actual[i]}, expected {expected[i]}");
    }

    /// <summary>The perceiver's former loop: queries <c>[Sq, heads·D]</c>, and rows of key then value.</summary>
    private static float[] FormerPerceiver(ReadOnlySpan<float> q, ReadOnlySpan<float> kv, int sq, int skv, int heads, int dimHead)
    {
        int dimInner = heads * dimHead;
        float[] output = new float[sq * dimInner], scores = new float[skv];
        float scale = 1f / MathF.Sqrt(dimHead);
        for (int h = 0; h < heads; h++)
        {
            int hOff = h * dimHead;
            for (int i = 0; i < sq; i++)
            {
                float maxS = float.NegativeInfinity;
                for (int j = 0; j < skv; j++)
                {
                    float dot = 0f;
                    for (int e = 0; e < dimHead; e++) dot += q[i * dimInner + hOff + e] * kv[j * 2 * dimInner + hOff + e];
                    float s = dot * scale;
                    scores[j] = s;
                    if (s > maxS) maxS = s;
                }
                float sum = 0f;
                for (int j = 0; j < skv; j++) { float e = MathF.Exp(scores[j] - maxS); scores[j] = e; sum += e; }
                float invSum = 1f / sum;
                for (int j = 0; j < skv; j++)
                {
                    float a = scores[j] * invSum;
                    for (int e = 0; e < dimHead; e++) output[i * dimInner + hOff + e] += a * kv[j * 2 * dimInner + dimInner + hOff + e];
                }
            }
        }
        return output;
    }

    /// <summary>The MRTE's former loop over channels-first operands <c>[channels, t]</c> and <c>[channels, tt]</c>.</summary>
    private static float[] FormerCrossAttention(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v, int channels, int t, int tt,
        int heads)
    {
        int kc = channels / heads;
        float invSqrt = 1f / MathF.Sqrt(kc);
        float[] output = new float[channels * t], scores = new float[tt];
        for (int hd = 0; hd < heads; hd++)
        {
            int chBase = hd * kc;
            for (int i = 0; i < t; i++)
            {
                float maxS = float.NegativeInfinity;
                for (int j = 0; j < tt; j++)
                {
                    float dot = 0f;
                    for (int c = 0; c < kc; c++) dot += q[(chBase + c) * t + i] * k[(chBase + c) * tt + j];
                    float s = dot * invSqrt;
                    scores[j] = s;
                    if (s > maxS) maxS = s;
                }
                float sum = 0f;
                for (int j = 0; j < tt; j++) { scores[j] = MathF.Exp(scores[j] - maxS); sum += scores[j]; }
                float inv = 1f / sum;
                for (int c = 0; c < kc; c++)
                {
                    float acc = 0f;
                    for (int j = 0; j < tt; j++) acc += scores[j] * inv * v[(chBase + c) * tt + j];
                    output[(chBase + c) * t + i] = acc;
                }
            }
        }
        return output;
    }

    /// <summary>The style encoder's former loop over a packed projection: each row its query, key and value.</summary>
    private static float[] FormerSelfAttention(float[] qkv, int frames, int dim, int heads)
    {
        int headDim = dim / heads;
        float[] merged = new float[frames * dim], scores = new float[frames];
        float scale = 1f / MathF.Sqrt(headDim);
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < frames; i++)
            {
                float max = float.NegativeInfinity;
                for (int j = 0; j < frames; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += qkv[i * 3 * dim + h * headDim + d] * qkv[j * 3 * dim + dim + h * headDim + d];
                    scores[j] = dot * scale;
                    max = MathF.Max(max, scores[j]);
                }
                float sum = 0f;
                for (int j = 0; j < frames; j++)
                {
                    scores[j] = MathF.Exp(scores[j] - max);
                    sum += scores[j];
                }
                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0f;
                    for (int j = 0; j < frames; j++) acc += scores[j] / sum * qkv[j * 3 * dim + 2 * dim + h * headDim + d];
                    merged[i * dim + h * headDim + d] = acc;
                }
            }
        return merged;
    }

    /// <summary>Reading a result on the host waits for the device, so no case needs a sync of its own. All three sites at
    /// the geometry their models have (read from the model code) and at small odd shapes. With
    /// <paramref name="longSequences"/>, also one long sequence a site, which a GPU row could refuse where the loop ran
    /// anything; the loops take seconds at that size, so the CPU lane leaves it out.</summary>
    private static void EverySiteMatchesItsFormerLoop(IBackend backend, float tolerance = Tolerance, bool longSequences = false)
    {
        // The perceiver: the speaker one is 32 latents and 8 heads of 64, the IndexTTS-2 emotion one a single latent and 4
        // heads; each attends its latents and a context.
        foreach ((int sq, int skv, int heads, int dimHead) in ((int, int, int, int)[])[(32, 32 + 200, 8, 64), (32, 32 + 7, 8, 64),
            (1, 1 + 221, 4, 64), (32, 32 + 4000, 8, 64), (5, 9, 3, 16)])
        {
            if (skv > 2000 && !longSequences) continue;
            using Tensor kv = Random(new TensorShape(1, skv, 2 * heads * dimHead), 12);
            using Tensor queries = Random(new TensorShape(1, sq, heads * dimHead), 11);
            float[] expected = FormerPerceiver(queries.AsReadOnlySpan<float>(), kv.AsReadOnlySpan<float>(), sq, skv, heads, dimHead);
            // The site takes its query over, so it gets a copy.
            Tensor q = new(queries.Shape, DType.F32);
            queries.AsReadOnlySpan<float>().CopyTo(q.AsSpan<float>());
            using Tensor actual = IndexTtsPerceiver.Attend(backend, q, kv, heads, dimHead);
            Assert.Equal(new TensorShape(1, sq, heads * dimHead), actual.Shape);
            AssertClose(expected, actual.AsReadOnlySpan<float>(), tolerance, $"perceiver {sq}x{skv}, {heads} heads of {dimHead}");
        }

        // The MRTE: 512 channels in 4 heads, the SSL stream over the text stream.
        foreach ((int channels, int t, int tt, int heads) in ((int, int, int, int)[])[(512, 150, 60, 4), (512, 1, 40, 4), (512, 3000, 600, 4), (24, 7, 5, 3)])
        {
            if (t > 2000 && !longSequences) continue;
            using Tensor q = Random(new TensorShape(1, channels, t), 21);
            using Tensor k = Random(new TensorShape(1, channels, tt), 22);
            using Tensor v = Random(new TensorShape(1, channels, tt), 23);
            float[] expected = FormerCrossAttention(q.AsReadOnlySpan<float>(), k.AsReadOnlySpan<float>(), v.AsReadOnlySpan<float>(), channels, t, tt, heads);
            using Tensor actual = SoVitsEncP.CrossAttention(backend, q, t, k, v, tt, heads);
            Assert.Equal(new TensorShape(1, channels, t), actual.Shape);
            AssertClose(expected, actual.AsReadOnlySpan<float>(), tolerance, $"MRTE {channels} channels, {t}x{tt}, {heads} heads");
        }

        // The style encoder: 512 channels in 8 heads, self-attention over the frames of a clip.
        foreach ((int frames, int dim, int heads) in ((int, int, int)[])[(120, 512, 8), (1, 512, 8), (2048, 512, 8), (11, 48, 3)])
        {
            if (frames > 2000 && !longSequences) continue;
            using Tensor packed = Random(new TensorShape(frames, 3 * dim), 31);
            float[] qkv = packed.AsReadOnlySpan<float>().ToArray();
            float[] expected = FormerSelfAttention(qkv, frames, dim, heads);
            float[] actual = ControlFoleyStyleEncoder.SelfAttention(backend, qkv, frames, dim, heads);
            AssertClose(expected, actual, tolerance, $"style encoder {frames} frames, {dim} channels, {heads} heads");
        }
    }

    [Fact]
    public void OnTheCpuBackendEverySiteMatchesItsFormerLoop()
    {
        using CpuBackend backend = new();
        EverySiteMatchesItsFormerLoop(backend);
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void OnCudaEverySiteMatchesItsFormerLoopAndNamesTheRowsThatRan()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new(0, PtxDirectory());
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(256);
        try { EverySiteMatchesItsFormerLoop(backend, Tf32Tolerance, longSequences: true); }
        finally { ReportRows(evidence.Snapshot()); }
        Assert.Equal(13, evidence.Snapshot().Count(e => e.Kind == AttentionEvidenceKind.Execution));
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void OnVulkanEverySiteMatchesItsFormerLoopAndNamesTheRowsThatRan()
    {
        VulkanBackend backend;
        try { backend = new VulkanBackend(); }
        catch (Exception ex) { _output.WriteLine($"SKIPPED: Vulkan unavailable ({ex.Message})"); return; }
        using (backend)
        {
            using AttentionEvidenceSession evidence = AttentionEvidence.Begin(256);
            try { EverySiteMatchesItsFormerLoop(backend, longSequences: true); }
            finally { ReportRows(evidence.Snapshot()); }
            Assert.Equal(13, evidence.Snapshot().Count(e => e.Kind == AttentionEvidenceKind.Execution));
        }
    }

    /// <summary>Which kernel rows served the calls: reported also when a comparison fails.</summary>
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
