using System.Text.Json;
using HartsyInference.Cpu;
using HartsyInference.Engine;
using HartsyInference.Engine.Registry;
using HartsyInference.Tests.Common;
using HartsyInference.LLM.DeepSeekV41;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The DSpark draft head with real weights against the unmodified upstream <c>forward_spec</c> (structural mode), dumped by
/// <c>tests/python-reference/deepseek_v41/dump_real_dspark.py</c>. Set <c>DSV41_DSPARK_ORACLE</c> to its output directory.</summary>
/// <remarks>The oracle runs in structural mode (no GEMM-input or latent quantization), so the port is loaded with <c>QuantizeLatents</c> off. Gates were fixed before the first
/// comparison: the draft ids must match exactly, every stage's attention, feed-forward and output must agree to a relative L2 of 2e-3 (the backbone's block gate), the logits to
/// 2e-3 relative L2, and each confidence to 2e-3 absolute. The prompt is short, so the sliding window never wraps; that path is covered by the backbone's window tests.</remarks>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class DeepSeekV41DSparkTests
{
    private const double StageRelL2 = 2e-3;
    private const double LogitsRelL2 = 2e-3;
    private const double ConfidenceAbs = 2e-3;

    private readonly ITestOutputHelper _output;

    public DeepSeekV41DSparkTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Draft_Matches_The_Upstream_Forward_Spec_On_The_Real_Weights()
    {
        string dir = ModelResolver.Resolve(DeepSeekV41Catalog.Id, null, Modality.Text).LocalPath ?? Path.Combine(RepoPaths.ModelsRoot(), "llm", "deepseek-v4.1-flash");
        string? oracle = Environment.GetEnvironmentVariable("DSV41_DSPARK_ORACLE");
        if (!RealWeightGate.Require(_output.WriteLine, Path.Combine(dir, "config.json"), oracle is null ? "DSV41_DSPARK_ORACLE-unset" : Path.Combine(oracle, "meta.json"))) return;

        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(oracle!, "meta.json")));
        JsonElement meta = doc.RootElement;
        int[] prompt = meta.GetProperty("prompt").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        int decodeToken = meta.GetProperty("decode_token").GetInt32(), startPos = meta.GetProperty("start_pos").GetInt32();
        int[] expectedIds = meta.GetProperty("draft_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        string Dump(string name) => Path.Combine(oracle!, name);
        // the prompt and the decode step must fit; 64 covers the 6-token oracle
        int capacity = Math.Max(64, prompt.Length + 8);

        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(dir);
        using CpuBackend backend = new();
        DeepSeekV41DSpark dspark = DeepSeekV41DSpark.Load(backend, checkpoint, new DeepSeekV41LoadOptions(MaxTokens: capacity, QuantizeLatents: false));
        int dim = checkpoint.Config.HiddenSize, vocab = checkpoint.Config.VocabSize, targetsWidth = checkpoint.Config.DsparkTargetLayerIds.Count * dim;
        Assert.Equal(prompt.Length, ReadF32(Dump("main_hidden_prefill.f32")).Length / targetsWidth);

        DeepSeekV41DSparkState state = dspark.CreateState(capacity);
        dspark.Seed(ReadF32(Dump("main_hidden_prefill.f32")), prompt.Length, state);
        float[] decodeHidden = ReadF32(Dump("main_hidden_decode.f32"));
        List<string> stageFailures = [];
        dspark.Probe = (kind, stage, values) =>
        {
            float[] refStage = ReadF32(Dump($"mtp{stage}.{kind}.f32"));
            Assert.Equal(refStage.Length, values.Length);
            double rel = RelL2(values, refStage);
            _output.WriteLine($"stage {stage} {kind}: relL2 {rel:E2}");
            if (rel > StageRelL2) stageFailures.Add($"stage {stage} {kind} relL2 {rel:E3}");
        };
        DeepSeekV41DSparkDraft draft = dspark.Draft(decodeToken, decodeHidden, startPos, state);
        dspark.Probe = null;
        Assert.True(stageFailures.Count == 0, string.Join("; ", stageFailures));

        float[] refLogits = ReadF32(Dump("draft_logits.f32"));
        float[] refConfidence = ReadF32(Dump("confidence.f32"));
        Assert.Equal(refLogits.Length, draft.Logits.Length);
        double logitsRel = RelL2(draft.Logits, refLogits);
        double confidenceWorst = refConfidence.Zip(draft.Confidence, (a, b) => Math.Abs(a - b)).Max();
        _output.WriteLine($"DSpark draft ids {string.Join(",", draft.Ids)} (oracle {string.Join(",", expectedIds)}); logits relL2 {logitsRel:E2}; "
            + $"confidence max abs {confidenceWorst:E2} (vocab {vocab}, block {dspark.BlockSize}, stages {dspark.StageCount})");

        Assert.Equal(expectedIds, draft.Ids);
        Assert.True(logitsRel <= LogitsRelL2, $"draft logits relL2 {logitsRel:E3} > {LogitsRelL2:E1}");
        Assert.True(confidenceWorst <= ConfidenceAbs, $"confidence max abs {confidenceWorst:E3} > {ConfidenceAbs:E1}");
    }

    private static float[] ReadF32(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        float[] values = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, values.Length * 4);
        return values;
    }

    private static double RelL2(float[] a, float[] b)
    {
        double diff = 0, norm = 0;
        for (int i = 0; i < a.Length; i++) { double d = (double)a[i] - b[i]; diff += d * d; norm += (double)b[i] * b[i]; }
        return Math.Sqrt(diff / Math.Max(norm, 1e-30));
    }
}
