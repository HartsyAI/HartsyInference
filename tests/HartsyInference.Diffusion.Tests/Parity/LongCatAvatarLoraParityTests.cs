using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.Lora;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>The avatar's release LoRAs against the pinned reference (tests/python-reference/longcat_avatar/dump_longcat_avatar_lora.py @ 58eb336) on
/// the tiny avatar DiT, CPU F32. The reference writes both LoRA files; the engine loads them the way the recipe does (<see cref="LoraFile"/>,
/// <see cref="RecipeLoraMerge"/> with gated adapters, the 1.6 per-step schedule) and runs the same call in each state the reference's demo can put
/// the model in: no LoRA, the 1.5 single LoRA at 0.8 (merged into the weights by the reference, a gated term here) and the 1.6 mix at its
/// boundary steps. Skipped, and so not evidence, until HARTSY_LONGCAT_AVATAR_LORA_DIR names the dump directory.</summary>
[Trait("Category", "Integration")]
public sealed class LongCatAvatarLoraParityTests(ITestOutputHelper output)
{
    private const string EnvName = "HARTSY_LONGCAT_AVATAR_LORA_DIR";
    private const double Tolerance = 1e-5;

    /// <summary>How far, in relative RMS, a LoRA case must be from the run without one for its match to say anything about the LoRA.</summary>
    private const double SmallestEffect = 1e-2;

    [LongCatAvatarLoraFact]
    public void EveryLoraStateMatchesTheReference()
    {
        string directory = Environment.GetEnvironmentVariable(EnvName)!;
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "meta.json")));
        JsonElement root = document.RootElement;
        float[] bare = LongCatReferenceDump.Read(directory, document, "velocity.base");
        List<string> failures = [];
        int cases = 0;
        foreach (JsonProperty entry in root.GetProperty("cases").EnumerateObject())
        {
            JsonElement state = entry.Value;
            string path = state.GetProperty("path").GetString()!;
            float[] expected = LongCatReferenceDump.Read(directory, document, $"velocity.{entry.Name}");
            float[] actual = path switch
            {
                "base" => Run(directory, document, static (_, _) => []),
                "merge" => Run(directory, document, (files, gates) =>
                {
                    Assert.Equal(LongCatRecipeSupport.DmdMultiplier, state.GetProperty("multiplier").GetSingle());
                    return [new GatedLora(files.Dmd, gates.Dmd, LongCatRecipeSupport.DmdMultiplier)];
                }, static gates => [gates.Dmd.Enable(1f)]),
                "mix" => Run(directory, document,
                    static (files, gates) => [new GatedLora(files.Dmd, gates.Dmd, 1f), new GatedLora(files.Opsd, gates.Opsd, 1f)],
                    gates =>
                    {
                        LongCatAvatarLoraSchedule schedule = new(gates.Dmd, gates.Opsd, state.GetProperty("alpha").GetSingle(),
                            state.GetProperty("mix_steps").GetInt32());
                        int step = state.GetProperty("step").GetInt32();
                        return [schedule.EnterDmd(step), schedule.EnterOpsd(step)];
                    }),
                _ => throw new InvalidOperationException($"Unknown case path '{path}'."),
            };
            double error = LongCatTestData.RelRms(actual, expected), effect = LongCatTestData.RelRms(expected, bare);
            output.WriteLine(FormattableString.Invariant($"{entry.Name,-16} relRMS {error:E3} (bound {Tolerance:E0}); from no LoRA {effect:E3}"));
            if (error > Tolerance)
                failures.Add($"{entry.Name}: velocity relRMS {error:E3} exceeds {Tolerance:E0}.");
            if (path != "base" && effect < SmallestEffect)
                failures.Add($"{entry.Name}: only {effect:E3} from the run without a LoRA.");
            cases++;
        }
        Assert.Equal(6, cases);
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    private sealed record Files(LoraFile Dmd, LoraFile Opsd);

    private sealed record Gates(AdjunctGate Dmd, AdjunctGate Opsd);

    /// <summary>One ai2v forward on a freshly loaded model: the adapters merged into the weights before the DiT takes them, the gates entered for
    /// the forward.</summary>
    private static float[] Run(string directory, JsonDocument document, Func<Files, Gates, GatedLora[]> attach,
        Func<Gates, IDisposable?[]>? enter = null)
    {
        JsonElement root = document.RootElement;
        LongCatConfig config = LongCatAvatarDitParityTests.TinyAvatarConfig();
        Dictionary<string, Tensor> weights = [];
        try
        {
            foreach (JsonProperty tensor in root.GetProperty("tensors").EnumerateObject())
                if (tensor.Name.StartsWith("dit.", StringComparison.Ordinal))
                    weights[tensor.Name["dit.".Length..]] = LongCatReferenceDump.Load(directory, document, tensor.Name);

            using CpuBackend backend = new();
            using LoraFile dmd = LoraFile.Load(Path.Combine(directory, "loraA.safetensors"));
            using LoraFile opsd = LoraFile.Load(Path.Combine(directory, "loraB.safetensors"));
            Gates gates = new(new AdjunctGate(), new AdjunctGate());
            GatedLora[] adapters = attach(new Files(dmd, opsd), gates);
            using IDisposable? stack = adapters.Length == 0 ? null
                : RecipeLoraMerge.Apply((IReadOnlyList<LoraResolver.LoraSpec>?)null, backend,
                    new LoraMergeTargets { Transformer = weights }, "LongCat", adapters);
            using LongCatTransformer transformer = new(config);
            transformer.LoadWeights(weights);

            int frames = root.GetProperty("frames").GetInt32(), height = root.GetProperty("h").GetInt32(), width = root.GetProperty("w").GetInt32();
            int textLength = root.GetProperty("text_len").GetInt32();
            using Tensor latent = Shaped(directory, document, "inputs.latent", 1, config.InChannels, frames, height, width);
            using Tensor encoder = Shaped(directory, document, "inputs.encoder", textLength, config.CaptionChannels);
            using Tensor audio = Shaped(directory, document, "inputs.audio", 1 + config.AudioVaeScale * (frames - 1),
                config.AudioWindow * config.AudioBlock * config.AudioChannel);
            IDisposable?[] scopes = enter?.Invoke(gates) ?? [];
            try
            {
                using Tensor tokens = transformer.ProjectAudio(backend, audio, frames);
                LongCatConditioning conditioning = new(NumCondLatents: 1);
                using Tensor velocity = transformer.Forward(backend, latent, encoder, [500f], in conditioning, audioTokens: tokens);
                return velocity.AsReadOnlySpan<float>().ToArray();
            }
            finally
            {
                foreach (IDisposable? scope in scopes)
                    scope?.Dispose();
            }
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }

    private static Tensor Shaped(string directory, JsonDocument document, string name, params long[] shape)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32);
        LongCatReferenceDump.Read(directory, document, name).CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private sealed class LongCatAvatarLoraFactAttribute : FactAttribute
    {
        public LongCatAvatarLoraFactAttribute()
        {
            string? directory = Environment.GetEnvironmentVariable(EnvName);
            if (string.IsNullOrWhiteSpace(directory) || !File.Exists(Path.Combine(directory, "meta.json")))
                Skip = $"SKIPPED: {EnvName} is unset or has no meta.json; the LongCat avatar LoRA reference comparison did not run.";
        }
    }
}
