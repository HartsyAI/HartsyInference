using System.Text.Json;
using HartsyInference.Cpu;
using HartsyInference.Engine;
using HartsyInference.Engine.Registry;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The C# target's DSpark taps on the real checkpoint, and the draft chained from them, against the upstream oracle written by <c>dump_real_dspark.py</c>.
/// Set <c>DSV41_DSPARK_ORACLE</c> to its output directory. The target runs the oracle's 6-token prompt and decode step in structural mode.</summary>
/// <remarks>Gates are fixed before the first comparison: each tapped row (prefill and decode) must agree with upstream's <c>main_hidden</c> to a relative L2 of 2e-3, the draft ids
/// must match exactly, and the logits and confidences meet the bounds of <see cref="DeepSeekV41DSparkTests"/>. The prompt is shorter than the window, so the window never wraps here.</remarks>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class DeepSeekV41DSparkChainTests
{
    private const double TapRelL2 = 2e-3;
    private const double LogitsRelL2 = 2e-3;
    private const double ConfidenceAbs = 2e-3;

    private readonly ITestOutputHelper _output;

    public DeepSeekV41DSparkChainTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Target_Taps_And_Draft_Chain_Match_The_Upstream_Oracle()
    {
        string dir = ModelResolver.Resolve(DeepSeekV41Catalog.Id, null, Modality.Text).LocalPath ?? Path.Combine(RepoPaths.ModelsRoot(), "llm", "deepseek-v4.1-flash");
        string? oracle = Environment.GetEnvironmentVariable("DSV41_DSPARK_ORACLE");
        if (!RealWeightGate.Require(_output.WriteLine, Path.Combine(dir, "config.json"), oracle is null ? "DSV41_DSPARK_ORACLE-unset" : Path.Combine(oracle, "meta.json"))) return;

        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(oracle!, "meta.json")));
        JsonElement meta = doc.RootElement;
        int[] prompt = meta.GetProperty("prompt").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        int decodeToken = meta.GetProperty("decode_token").GetInt32(), startPos = meta.GetProperty("start_pos").GetInt32();
        int[] targetLayers = meta.GetProperty("target_layers").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        int[] expectedIds = meta.GetProperty("draft_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        string Dump(string name) => Path.Combine(oracle!, name);

        using CpuBackend backend = new();
        using DeepSeekV41LoadedModel loaded = DeepSeekV41HostModelLoader.Load(backend, dir, new DeepSeekV41LoadOptions(MaxTokens: 64, QuantizeLatents: false));
        DeepSeekV41HostModel model = loaded.Model;
        int dim = loaded.Checkpoint.Config.HiddenSize, width = model.MainHiddenWidth;
        Assert.Equal(targetLayers, model.MainHiddenLayers);
        Assert.Equal(targetLayers.Length * dim, width);

        DeepSeekV41SequenceState state = model.CreateState(64);
        float[] prefillHidden = new float[prompt.Length * dim], prefillMain = new float[prompt.Length * width];
        model.Forward(prompt, state, prefillHidden, prefillMain);
        float[] refPrefill = ReadF32(Dump("main_hidden_prefill.f32"));
        Assert.Equal(refPrefill.Length, prefillMain.Length);
        double prefillRel = RelL2(prefillMain, refPrefill);

        float[] decodeHidden = new float[dim], decodeMain = new float[width];
        model.Forward(new[] { decodeToken }, state, decodeHidden, decodeMain);
        float[] refDecode = ReadF32(Dump("main_hidden_decode.f32"));
        Assert.Equal(refDecode.Length, decodeMain.Length);
        double decodeRel = RelL2(decodeMain, refDecode);
        _output.WriteLine($"target taps (layers {string.Join(",", targetLayers)}, width {width}): prefill relL2 {prefillRel:E2}, decode relL2 {decodeRel:E2}");
        Assert.True(prefillRel <= TapRelL2, $"prefill main_hidden relL2 {prefillRel:E3} > {TapRelL2:E1}");
        Assert.True(decodeRel <= TapRelL2, $"decode main_hidden relL2 {decodeRel:E3} > {TapRelL2:E1}");

        DeepSeekV41DSpark dspark = DeepSeekV41DSpark.Load(backend, loaded.Checkpoint, new DeepSeekV41LoadOptions(MaxTokens: 64, QuantizeLatents: false));
        DeepSeekV41DSparkState dstate = dspark.CreateState(64);
        dspark.Seed(prefillMain, prompt.Length, dstate);
        DeepSeekV41DSparkDraft draft = dspark.Draft(decodeToken, decodeMain, startPos, dstate);

        float[] refLogits = ReadF32(Dump("draft_logits.f32")), refConfidence = ReadF32(Dump("confidence.f32"));
        Assert.Equal(refLogits.Length, draft.Logits.Length);
        double logitsRel = RelL2(draft.Logits, refLogits);
        double confidenceWorst = refConfidence.Zip(draft.Confidence, (a, b) => Math.Abs(a - b)).Max();
        _output.WriteLine($"draft chain: ids {string.Join(",", draft.Ids)} (oracle {string.Join(",", expectedIds)}); logits relL2 {logitsRel:E2}; confidence max abs {confidenceWorst:E2}");
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
