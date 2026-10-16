using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Models.MuseSpark;
using HartsyInference.LLM.OutputParsing;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>Every llama.cpp quantization of the small random Muse Spark (<c>make_muse_spark_small_gguf.py</c>) on the CPU backend, against
/// transformers on that file's own dequantized weights (the exact answer) and llama.cpp's logits. llama.cpp rounds matmul activations to
/// 8 bits on quantized files, so it must match tightly only for F32/F16; elsewhere the engine must be at least as close to exact.</summary>
[Trait("Category", "Integration")]
public sealed unsafe class MuseSparkGgufParityTests(ITestOutputHelper output)
{
    private const float ExactTolerance = 2e-3f;

    /// <summary>Every quantization llama-quantize offers, plus the converter's F32.</summary>
    public static TheoryData<string> Quants => new()
    {
        "F32", "F16", "BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1", "Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1", "IQ4_NL", "IQ4_XS",
        "Q3_K_L", "Q3_K_M", "Q3_K_S", "IQ3_M", "IQ3_S", "IQ3_XS", "IQ3_XXS", "Q2_K", "Q2_K_S", "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS",
        "IQ1_M", "IQ1_S", "TQ1_0", "TQ2_0", "MXFP4_MOE", "Q1_0", "Q2_0",
    };

    [Theory]
    [MemberData(nameof(Quants))]
    public void Quant_MatchesDequantizedReferenceAndLlamaCpp(string quant)
    {
        string dir = TestPaths.Llm.MuseSparkSmallDir;
        string gguf = Path.Combine(dir, $"muse-spark-small-{quant}.gguf");
        if (!RealWeightGate.Require(output.WriteLine, gguf)) return;

        using SafeTensorsLoader reference = new();
        reference.Load(Path.Combine(dir, "transformers_reference.safetensors"));
        int[] ids = reference.GetTensor("input_ids").AsReadOnlySpan<int>().ToArray();
        using GgufLanguageModel model = GgufLanguageModel.Load(gguf, dequantizeToF32: true);
        using CpuBackend backend = new();
        using KvCache cache = new(model.Config.NumLayers, 1, model.Config.NumKvHeads, model.Config.HeadDim);
        using Tensor hidden = model.Transformer.Forward(backend, ids, 0, cache);
        using Tensor logits = model.Transformer.ProjectLogits(backend, hidden, ids.Length);
        ReadOnlySpan<float> got = logits.AsReadOnlySpan<float>();
        int vocab = model.Config.VocabSize;
        ReadOnlySpan<float> lastRow = got.Slice((ids.Length - 1) * vocab, vocab);

        string dequant = Path.ChangeExtension(gguf, ".dequant.safetensors");
        float[]? exactLast = null;
        if (File.Exists(dequant))
        {
            using SafeTensorsLoader exact = new();
            exact.Load(dequant);
            ReadOnlySpan<float> exactLogits = exact.GetTensor("logits").AsReadOnlySpan<float>();
            exactLast = exactLogits.Slice((ids.Length - 1) * vocab, vocab).ToArray();
            float d = MaxAbsDiff(got, exactLogits);
            output.WriteLine($"{quant}: vs transformers on the dequantized weights max |Δ| = {d:E3} over {ids.Length} rows");
            Assert.True(d < ExactTolerance, $"{quant}: {d} against the exact dequantized reference");
        }
        if (quant is "F32" or "F16" or "BF16")
        {
            float d = MaxAbsDiff(got, reference.GetTensor("logits").AsReadOnlySpan<float>());
            output.WriteLine($"{quant}: vs original transformers max |Δ| = {d:E3}");
        }

        string llama = Path.Combine(dir, "llamacpp", $"llamacpp-muse-spark-small-{quant}.bin");
        if (File.Exists(llama))
        {
            float[] llamaLogits = ReadFloats(llama);
            int[] llamaTokens = ReadInts(Path.Combine(dir, "llamacpp", $"llamacpp-muse-spark-small-{quant}-tokens.bin"));
            Assert.Equal(ids, llamaTokens);
            float d = MaxAbsDiff(lastRow, llamaLogits);
            int ourTop = ArgMax(lastRow), theirTop = ArgMax(llamaLogits);
            double kl = KlDivergence(llamaLogits, lastRow);
            output.WriteLine($"{quant}: vs llama.cpp last token max |Δ| = {d:E3}, KL(llama.cpp‖ours) = {kl:E3}, top-1 {ourTop} vs {theirTop}");
            if (quant is "F32" or "F16") Assert.True(kl < 1e-4, $"{quant}: KL {kl} against llama.cpp, which does not round these activations");
            if (exactLast is not null)
            {
                float ours = MaxAbsDiff(lastRow, exactLast), theirs = MaxAbsDiff(llamaLogits, exactLast);
                output.WriteLine($"{quant}: distance from the exact answer, ours {ours:E3} vs llama.cpp {theirs:E3}");
                Assert.True(ours <= Math.Max(theirs, ExactTolerance),
                    $"{quant}: ours {ours} is further from the exact answer than llama.cpp's {theirs}");
            }
        }
    }

    [Fact]
    public void Tokenizer_MatchesTransformersIds()
    {
        string dir = TestPaths.Llm.MuseSparkSmallDir;
        string gguf = Path.Combine(dir, "muse-spark-small-Q8_0.gguf");
        if (!RealWeightGate.Require(output.WriteLine, gguf)) return;
        using JsonDocument prompt = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "prompt.json")));
        int[] want = [.. prompt.RootElement.GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt32())];
        using GgufLanguageModel model = GgufLanguageModel.Load(gguf, dequantizeToF32: true);
        Assert.Equal(want, model.Tokenizer.Encode(prompt.RootElement.GetProperty("prompt").GetString()!, addSpecial: true));
    }

    [Fact]
    public void OutputParser_ReadsATurnInTheRealVocabulary()
    {
        string dir = TestPaths.Llm.MuseSparkSmallDir;
        string gguf = Path.Combine(dir, "muse-spark-small-Q8_0.gguf");
        if (!RealWeightGate.Require(output.WriteLine, gguf)) return;
        using GgufLanguageModel model = GgufLanguageModel.Load(gguf, dequantizeToF32: true);
        Assert.True(MuseSparkOutputParser.Handles(model.Template, model.Tokenizer));
        const string completion = " to=self<|message|>Paris 🇫🇷 is on CET.<|eom|><|start|>assistant to=get_time<|message|>"
            + "<atem:function_calls>\n<atem:invoke name=\"get_time\">\n<atem:parameter name=\"tz\">Europe/Paris</atem:parameter>\n"
            + "</atem:invoke>\n</atem:function_calls><|eot|>";
        int[] ids = model.Tokenizer.Encode(completion, addSpecial: true);
        Assert.Contains(model.Tokenizer.SpecialId("<|eom|>")!.Value, ids);

        MuseSparkOutputParser parser = new(model.Tokenizer);
        foreach (int id in ids) parser.Push(id, _ => { });
        parser.Finish(_ => { });

        Assert.Equal("Paris 🇫🇷 is on CET.", parser.Result.Reasoning);
        ChatToolCall call = Assert.Single(parser.Result.ToolCalls);
        Assert.Equal(("get_time", """{"tz":"Europe/Paris"}"""), (call.Name, call.ArgumentsJson));
        Assert.False(parser.Result.Malformed);
    }

    [Fact]
    public void GgufConfig_MatchesHuggingFaceConfig()
    {
        string dir = TestPaths.Llm.MuseSparkSmallDir;
        string gguf = Path.Combine(dir, "muse-spark-small-Q8_0.gguf");
        if (!RealWeightGate.Require(output.WriteLine, gguf)) return;
        TransformerConfig hf = MuseSparkHfConfig.FromJson(File.ReadAllText(Path.Combine(dir, "hf", "config.json"))).Transformer;
        using GgufLanguageModel model = GgufLanguageModel.Load(gguf, dequantizeToF32: true);
        Assert.Equal(Signature(hf), Signature(model.Config));
        // llama.cpp unpermutes Q/K for its interleaved rope; transformers keeps rotate_half.
        Assert.Equal(RopeStyle.SplitHalf, hf.Rope);
        Assert.Equal(RopeStyle.Interleaved, model.Config.Rope);
    }

    /// <summary>Every field the Muse Spark forward reads, as one comparable line.</summary>
    private static string Signature(TransformerConfig c)
    {
        MoeConfig m = c.Moe!;
        string layers = string.Concat(Enumerable.Range(0, c.NumLayers).Select(i => c.IsGlobalLayer(i) ? "G" : "L"));
        return string.Join(" ", c.HiddenSize, c.NumLayers, c.NumHeads, c.NumKvHeads, c.HeadDim, c.VocabSize, c.RopeThetaFor(0),
            c.RmsNormEps, c.EffPostNormEps, c.SlidingWindow, c.FinalLogitSoftcap, MathF.Round(1f / c.LogitScale, 6), layers, c.QkNorm,
            c.QkNormFullDim, c.SandwichNorm, c.NoRopeOnGlobalLayers, c.TieWordEmbeddings, c.EmbeddingRmsNorm, c.AttnHeadOutputNorm,
            c.AttnOutputGate, c.PostAttnNormWeightless, c.PostFfnNormScalesInput, c.ResidualGates, c.AttnScale, m.NumExperts,
            m.NumExpertsPerTok, m.MoeIntermediateSize, m.Scoring, m.NormTopKProb, m.RoutedScalingFactor, m.FirstDenseLayers,
            m.LatentSize, m.LatentNormEps, m.PostExpertNorm, m.PostExpertNormEps, m.SharedExpertIntermediateSize, m.ExpertGroupCount);
    }

    private static float MaxAbsDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        Assert.Equal(b.Length, a.Length);
        float worst = 0f;
        for (int i = 0; i < a.Length; i++) worst = MathF.Max(worst, MathF.Abs(a[i] - b[i]));
        return worst;
    }

    private static int ArgMax(ReadOnlySpan<float> x)
    {
        int best = 0;
        for (int i = 1; i < x.Length; i++) if (x[i] > x[best]) best = i;
        return best;
    }

    /// <summary>KL(p‖q) of the softmax distributions of two logit rows.</summary>
    private static double KlDivergence(ReadOnlySpan<float> pLogits, ReadOnlySpan<float> qLogits)
    {
        double pMax = double.NegativeInfinity, qMax = double.NegativeInfinity;
        for (int i = 0; i < pLogits.Length; i++) { pMax = Math.Max(pMax, pLogits[i]); qMax = Math.Max(qMax, qLogits[i]); }
        double pSum = 0, qSum = 0;
        for (int i = 0; i < pLogits.Length; i++) { pSum += Math.Exp(pLogits[i] - pMax); qSum += Math.Exp(qLogits[i] - qMax); }
        double pLog = Math.Log(pSum), qLog = Math.Log(qSum), kl = 0;
        for (int i = 0; i < pLogits.Length; i++)
        {
            double lp = pLogits[i] - pMax - pLog, lq = qLogits[i] - qMax - qLog;
            kl += Math.Exp(lp) * (lp - lq);
        }
        return kl;
    }

    private static float[] ReadFloats(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        float[] values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, values.Length * sizeof(float));
        return values;
    }

    private static int[] ReadInts(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        int[] values = new int[bytes.Length / sizeof(int)];
        Buffer.BlockCopy(bytes, 0, values, 0, values.Length * sizeof(int));
        return values;
    }
}
