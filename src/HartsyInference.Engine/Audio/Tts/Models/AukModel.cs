using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Auk;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>Tencent AuK (<c>auk:base</c>, 32-step guided) and AuK-Flash (<c>auk:flash</c>, 4 steps, no guidance): an instruction-driven flow-matching DiT conditioned by the Qwen2.5-Omni-3B thinker, vocoded by the AuK VAE to 24 kHz. Zero-shot cloning needs a reference clip; without one the instruction describes the voice and the duration must be given. The Qwen2.5-Omni encoder is under the Qwen Research License.</summary>
internal static class AukModel
{
    internal const string AukRepo = "tencent/AuK";
    internal const string FlashRepo = "tencent/AuK-Flash";
    internal const string OmniRepo = "Qwen/Qwen2.5-Omni-3B";
    internal const string BaseFile = "auk_base.safetensors";
    internal const string FlashFile = "auk_flash.safetensors";
    internal const string VaeFile = "vae.safetensors";
    internal const string TokenizerFile = "tokenizer.json";

    /// <summary>The thinker text LM and the audio tower are the only Omni tensors AuK reads; the talker, vision tower and token2wav stay on the hub.</summary>
    internal static readonly string[] OmniKeyPrefixes = ["thinker.model.", "thinker.audio_tower."];

    internal const string TokenizerSha256 = "8441917e39ae0244e06d704b95b3124795cec478e297f9afac39ba670d7e9d99";

    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = ResolveRepo,
        ResolveFiles = async (variant, cancel) =>
        {
            List<AudioModelFile> files = [];
            foreach (AudioModelFile omni in await AudioCheckpoints.ResolveCheckpointFilesAsync(OmniRepo, "tts", cancel, OmniKeyPrefixes).ConfigureAwait(false))
            {
                files.Add(omni with { Repo = OmniRepo });
            }
            files.Add(new AudioModelFile(TokenizerFile, Sha256: TokenizerSha256, Repo: OmniRepo));
            files.Add(new AudioModelFile(VaeFile, Repo: AukRepo));
            files.Add(new AudioModelFile(CheckpointFile(variant)));
            return files;
        },
        LoadAsync = async (_, variant, cancel) =>
        {
            bool flash = IsFlash(variant);
            string repo = ResolveRepo(variant);
            (IReadOnlyDictionary<string, Tensor> omni, IDisposable[] omniLoaders) =
                await AudioCheckpoints.LoadAsync(OmniRepo, "tts", cancel, OmniKeyPrefixes).ConfigureAwait(false);
            IReadOnlyDictionary<string, string> fetched = await AudioModelCache.FetchAllAsync(repo,
                [new AudioModelFile(TokenizerFile, Sha256: TokenizerSha256, Repo: OmniRepo), new AudioModelFile(VaeFile, Repo: AukRepo), new AudioModelFile(CheckpointFile(variant))],
                "tts", ct: cancel).ConfigureAwait(false);

            SafeTensorsLoader aukLoader = new SafeTensorsLoader();
            aukLoader.Load(fetched[CheckpointFile(variant)]);
            SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
            vaeLoader.Load(fetched[VaeFile]);
            GgufTokenizer tokenizer;
            using (FileStream stream = File.OpenRead(fetched[TokenizerFile]))
            {
                tokenizer = HfTokenizerJson.LoadByteLevelBpe(stream);
            }

            AukPipeline pipeline = new AukPipeline(repo, flash, tokenizer.EncodeOrdinary);
            pipeline.LoadWeights(aukLoader.GetAllTensors(), vaeLoader.GetAllTensors(), omni);
            Logs.Info($"[Audio][AuK] Loaded {repo} ({(flash ? "Flash: 4 steps, no guidance" : "base: 32 steps, guided")}; Qwen2.5-Omni-3B encoder, 24 kHz).");

            IDisposable?[] keep = [pipeline, aukLoader, vaeLoader, .. omniLoaders];
            return new TtsRunner(pipeline.SampleRate, (backend, job) => Synthesize(pipeline, backend, job), keep);
        },
    };

    /// <summary>True for the distilled Flash variant (<c>flash</c> or a Flash repo id).</summary>
    internal static bool IsFlash(string? variant) => (variant ?? string.Empty).Contains("flash", StringComparison.OrdinalIgnoreCase);

    /// <summary>Maps the variant hint to the repo holding the DiT checkpoint; a custom <c>owner/name</c> passes through.</summary>
    internal static string ResolveRepo(string? variant)
    {
        string id = (variant ?? string.Empty).Trim();
        if (id.Contains('/', StringComparison.Ordinal))
        {
            return id;
        }
        return IsFlash(id) ? FlashRepo : AukRepo;
    }

    /// <summary>The DiT checkpoint file name of the variant.</summary>
    internal static string CheckpointFile(string? variant) => IsFlash(variant) ? FlashFile : BaseFile;

    /// <summary>Resolves the ChatML instruction: an explicit one is verbatim with a reference clip and frames the text as a voice description without one; no instruction means zero-shot cloning.</summary>
    internal static string BuildInstruction(string text, string? instruction, bool hasReference)
    {
        bool hasInstruction = !string.IsNullOrWhiteSpace(instruction);
        if (hasReference)
        {
            return hasInstruction ? instruction! : AukTemplates.Format(AukTask.ZeroShotTts, text);
        }
        if (!hasInstruction)
        {
            throw new InvalidOperationException(
                "AuK needs a voice reference clip (zero-shot cloning) or an instruction describing the voice (--instruction).");
        }
        return string.IsNullOrWhiteSpace(text) ? instruction! : AukTemplates.Format(AukTask.InstructTts, instruction!, text);
    }

    private static float[] Synthesize(AukPipeline pipeline, HartsyInference.Core.Backends.IBackend backend, TtsJob job)
    {
        bool hasReference = job.ReferenceMono24k is { Length: > 0 };
        string instruction = BuildInstruction(job.Text, job.Instruction, hasReference);
        AukOptions options = new AukOptions
        {
            Seed = job.Seed,
            Steps = job.NfeStep,
            CfgScale = job.CfgScale.HasValue ? (float)job.CfgScale.Value : null,
            Speed = job.Speed ?? 1.0,
            GenText = job.Text,
            RefText = job.RefText,
        };
        return pipeline.Generate(backend, instruction, job.ReferenceMono24k, pipeline.SampleRate, job.DurationSeconds, options, job.Cancel);
    }
}
