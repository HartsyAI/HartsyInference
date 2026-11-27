using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Attention;

/// <summary>The HTS-AT Swin stage's window attention through the shared attention layer, against the host loop the stage
/// ran before: per window and head, scores <c>q·k·scale + bias + mask</c>, where the bias is the relative-position term
/// and the mask is the shifted block's 0 or -100. The loop is kept here, verbatim in its arithmetic, as the oracle.
/// <para>The tolerances were fixed before the first run: 1e-4 relative to the larger of 1 and the value on the CPU backend
/// and on Vulkan, and on CUDA the contract bound for TF32, 2e-3.</para></summary>
[Collection("AttentionEvidence")]
public sealed class SwinWindowAttentionTests
{
    private const float Tolerance = 1e-4f, Tf32Tolerance = 2e-3f;

    private readonly ITestOutputHelper _output;

    public SwinWindowAttentionTests(ITestOutputHelper output) => _output = output;

    private static float[] FormerLoop(float[] qkv, float[] bias, float[]? mask, int windows, int tokens, int c, int heads)
    {
        int headDim = c / heads;
        float scale = MathF.Pow(headDim, -0.5f);
        float[] merged = new float[windows * tokens * c], scores = new float[tokens];
        for (int w = 0; w < windows; w++)
        {
            int baseRow = w * tokens;
            for (int h = 0; h < heads; h++)
                for (int i = 0; i < tokens; i++)
                {
                    int qOff = (baseRow + i) * 3 * c + h * headDim;
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < tokens; j++)
                    {
                        int kOff = (baseRow + j) * 3 * c + c + h * headDim;
                        float dot = 0f;
                        for (int d = 0; d < headDim; d++) dot += qkv[qOff + d] * scale * qkv[kOff + d];
                        dot += bias[(h * tokens + i) * tokens + j];
                        if (mask is not null) dot += mask[(w * tokens + i) * tokens + j];
                        scores[j] = dot;
                        max = MathF.Max(max, dot);
                    }
                    float sum = 0f;
                    for (int j = 0; j < tokens; j++)
                    {
                        scores[j] = MathF.Exp(scores[j] - max);
                        sum += scores[j];
                    }
                    int outOff = (baseRow + i) * c + h * headDim;
                    for (int d = 0; d < headDim; d++)
                    {
                        float acc = 0f;
                        for (int j = 0; j < tokens; j++) acc += scores[j] / sum * qkv[(baseRow + j) * 3 * c + 2 * c + h * headDim + d];
                        merged[outOff + d] = acc;
                    }
                }
        }
        return merged;
    }

    /// <summary>The four stages of the audio branch (window 8 over grids of 64, 32, 16 and 8; widths 128 to 1,024; 4 to 32
    /// heads), each with and without the shift mask except the last, whose one window is never shifted, and a small odd
    /// shape. A stage's shifted case uses the mask the stage itself builds; the odd shape uses a random one.</summary>
    private static void MatchesTheFormerLoop(IBackend backend, float tolerance)
    {
        foreach ((int windows, int tokens, int dim, int heads, bool shifted) in ((int, int, int, int, bool)[])[(64, 64, 128, 4, false),
            (64, 64, 128, 4, true), (16, 64, 256, 8, false), (16, 64, 256, 8, true), (4, 64, 512, 16, false), (4, 64, 512, 16, true),
            (1, 64, 1024, 32, false), (3, 5, 12, 3, true)])
        {
            Random random = new(windows * 131 + dim + (shifted ? 1 : 0));
            float[] qkv = new float[windows * tokens * 3 * dim], bias = new float[heads * tokens * tokens];
            for (int i = 0; i < qkv.Length; i++) qkv[i] = random.NextSingle() * 2f - 1f;
            for (int i = 0; i < bias.Length; i++) bias[i] = random.NextSingle() * 2f - 1f;
            float[]? mask = null;
            if (shifted)
            {
                if (tokens == 64)
                {
                    // The stage's own shift mask: its grid is the window (8) times the windows a side, shifted by half a window.
                    mask = ControlFoleySwinStage.BuildShiftMask(8 * (int)Math.Sqrt(windows), 8, 4);
                    Assert.Equal(windows * tokens * tokens, mask.Length);
                    Assert.Contains(-100f, mask);
                }
                else
                {
                    mask = new float[windows * tokens * tokens];
                    for (int i = 0; i < mask.Length; i++) mask[i] = random.Next(4) == 0 ? -100f : 0f;
                }
            }

            float[] expected = FormerLoop(qkv, bias, mask, windows, tokens, dim, heads);
            float[] actual = ControlFoleySwinStage.AttendWindows(backend, qkv,
                ControlFoleySwinStage.CombineTerms(bias, mask, windows, tokens, heads), windows, tokens, dim, heads);
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1f, Math.Abs(expected[i])),
                    $"{windows} windows of {tokens}, {dim} channels, {heads} heads, shifted {shifted}: element {i} is {actual[i]}, expected {expected[i]}");
        }
    }

    [Fact]
    public void OnTheCpuBackendItMatchesTheFormerLoop()
    {
        using CpuBackend backend = new();
        MatchesTheFormerLoop(backend, Tolerance);
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void OnCudaItMatchesTheFormerLoopAndNamesTheRowsThatRan()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new(0, PtxDirectory());
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(256);
        try { MatchesTheFormerLoop(backend, Tf32Tolerance); }
        finally { ReportRows(evidence.Snapshot()); }
        Assert.Equal(8, evidence.Snapshot().Count(e => e.Kind == AttentionEvidenceKind.Execution));
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
            try { MatchesTheFormerLoop(backend, Tolerance); }
            finally { ReportRows(evidence.Snapshot()); }
            Assert.Equal(8, evidence.Snapshot().Count(e => e.Kind == AttentionEvidenceKind.Execution));
        }
    }

    /// <summary>Which kernel rows served the calls, and what else was recorded: reported also when a comparison fails.</summary>
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
