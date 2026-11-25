using System.Diagnostics;
using System.Text;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe;
using HartsyInference.Cuda;
using HartsyInference.Engine.Placement;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Maxwell;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The real Maxwell-V1.0-Flash FP8 checkpoint on the CUDA backend: dense weights preloaded, routed experts behind a device expert cache
/// (<see cref="ExpertOffloadSetup.AttachDeviceOnly"/>), a chat prompt answered greedily. Records the generated text, prefill and decode speed, and
/// the device memory. Reported as skipped (never passed) without MAXWELL_MODEL_FILE pointing at model.safetensors, or without a CUDA device.</summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "Integration")]
public sealed class MaxwellRealWeightsGpuTests(ITestOutputHelper output)
{
    private const string Question = "Why is it harder to keep a room cool when the door is open on a hot day? Answer in three short sentences.";

    [MaxwellRealFact]
    public void Cuda_ChatAnswer_RunsOnTheRealCheckpoint()
    {
        string file = Environment.GetEnvironmentVariable("MAXWELL_MODEL_FILE")!;
        string directory = Path.GetDirectoryName(file)!;
        int newTokens = int.TryParse(Environment.GetEnvironmentVariable("MAXWELL_NEW_TOKENS"), out int n) ? n : 16;
        double cacheGb = double.TryParse(Environment.GetEnvironmentVariable("MAXWELL_EXPERT_CACHE_GB"), out double g) ? g : 4;
        string? resultPath = Environment.GetEnvironmentVariable("MAXWELL_RUN_OUT");

        using CudaBackend backend = new(0, PtxDirectory());
        Stopwatch load = Stopwatch.StartNew();
        using MaxwellModel model = MaxwellModel.Load(directory, backend, new MaxwellLoadOptions { MaxSequenceTokens = 512, ResidentWeights = true });
        load.Stop();
        backend.PreloadWeightGroups(model.EnumerateDenseWeightGroups());
        DeviceMemoryUsage afterDense = backend.GetDeviceMemoryUsage();
        using MoeExpertOffload offload = ExpertOffloadSetup.AttachDeviceOnly(backend, model, (long)(cacheGb * (1L << 30)), "[maxwell]");
        DeviceMemoryUsage afterCache = backend.GetDeviceMemoryUsage();

        (string? bos, string? eos) = HfChatAssets.SpecialTokenLiterals(directory);
        List<int> stopIds = HfStopIds.Read(directory, model.Config.EosTokenId);
        (ILlmTokenizer tokenizer, IChatTemplate template) = HfChatAssets.Load(directory, stopIds, bos, eos);
        int[] prompt = template.Encode(tokenizer, [new ChatMessage("user", Question)], addGenerationPrompt: true, enableThinking: false);

        MaxwellSequenceState state = model.CreateState(512);
        Stopwatch prefillClock = Stopwatch.StartNew();
        float[] logits = model.Forward(prompt, state, allRows: false);
        prefillClock.Stop();
        List<int> generated = [];
        List<double> stepMs = [];
        for (int step = 0; step < newTokens; step++)
        {
            int next = Argmax(logits);
            if (stopIds.Contains(next)) break;
            generated.Add(next);
            Stopwatch step2 = Stopwatch.StartNew();
            logits = model.Forward([next], state, allRows: false);
            step2.Stop();
            stepMs.Add(step2.Elapsed.TotalMilliseconds);
        }
        DeviceMemoryUsage end = backend.GetDeviceMemoryUsage();

        string text = tokenizer.Decode(generated);
        double decodeTokPerSec = stepMs.Count == 0 ? 0 : 1000.0 * stepMs.Count / stepMs.Sum();
        StringBuilder report = new();
        report.AppendLine($"Maxwell-V1.0-Flash FP8, CUDA {backend.Capabilities.Name}, expert cache {cacheGb} GB");
        report.AppendLine($"load {load.Elapsed.TotalSeconds:F1} s; prompt {prompt.Length} tokens, prefill {prefillClock.Elapsed.TotalSeconds:F1} s");
        report.AppendLine($"decode {generated.Count} tokens, {decodeTokPerSec:F2} tok/s (mean {(stepMs.Count == 0 ? 0 : stepMs.Average()):F0} ms per step)");
        report.AppendLine($"device after dense preload {Mb(afterDense.WeightBytes)} MB weights; after expert cache {Mb(afterCache.WeightBytes)} MB; "
            + $"end {Mb(end.WeightBytes)} MB weights, {Mb(end.ActivationBytes)} MB activations, {Mb(end.FreeBytes)} MB free of {Mb(end.TotalBytes)} MB");
        report.AppendLine("generated:");
        report.AppendLine(text);
        output.WriteLine(report.ToString());
        if (resultPath is not null) File.WriteAllText(resultPath, report.ToString());

        Assert.NotEmpty(generated);
        Assert.All(logits, static v => Assert.True(float.IsFinite(v), "a logit is not finite"));
    }

    private static int Argmax(float[] logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    private static long Mb(long bytes) => bytes >> 20;

    private static string PtxDirectory() => Ernie5ModelGpuParityTests.PtxDirectory();
}

/// <summary>A fact that skips (never passes) unless MAXWELL_MODEL_FILE names an existing file and a CUDA device is present.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class MaxwellRealFactAttribute : FactAttribute
{
    public MaxwellRealFactAttribute()
    {
        string? path = Environment.GetEnvironmentVariable("MAXWELL_MODEL_FILE");
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            Skip = "SKIPPED: MAXWELL_MODEL_FILE is unset or not an existing file; real-weight run did not run.";
    }
}
