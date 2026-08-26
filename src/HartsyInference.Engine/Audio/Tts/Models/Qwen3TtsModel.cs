using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.QwenTts;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>Qwen3-TTS (Qwen/Qwen3-TTS-12Hz-*) — 12 Hz talker (semantic codebook-0) + MTP code predictor (codebooks 1..15) + Snake/ConvNeXt vocoder → 24 kHz. The variant hint encodes the size and mode: <c>*-CustomVoice</c> → preset speakers, <c>*-VoiceDesign</c> → instruct text, anything else → the Base checkpoint's voice clone.</summary>
internal static class Qwen3TtsModel
{
    private const string DefaultRepo = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice";

    /// <summary>English preset speaker id on the CustomVoice checkpoint (codec space).</summary>
    private const int EnglishSpeakerToken = 3061;

    /// <summary>English codec-space language id (LanguageIdBase 2050) used to condition voice_clone.</summary>
    private const int EnglishLanguageId = 2050;

    /// <summary>Shared by the loader and prefetch. The codec is fetched before the talker weights that mark
    /// the model installed.</summary>
    private static readonly AudioModelFile[] ModelFiles =
    [
        new("speech_tokenizer/model.safetensors"),
        new("model.safetensors"),
    ];

    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = ResolveRepo,
        ResolveFiles = (_, _) => Task.FromResult<IReadOnlyList<AudioModelFile>>(ModelFiles),
        LoadAsync = async (_, variant, cancel) =>
        {
            string repo = ResolveRepo(variant);
            string mode = ResolveMode(variant);
            bool small = repo.Contains("0.6B", StringComparison.OrdinalIgnoreCase);

            // Two checkpoints: model.safetensors carries talker.* (+ MTP under talker.code_predictor.*) and
            // speaker_encoder.* (ECAPA); speech_tokenizer/model.safetensors carries the codec.
            IReadOnlyDictionary<string, string> fetched = await AudioModelCache
                .FetchAllAsync(repo, ModelFiles, category: "tts", ct: cancel).ConfigureAwait(false);
            string talkerPath = fetched["model.safetensors"];
            string codecPath = fetched["speech_tokenizer/model.safetensors"];
            SafeTensorsLoader talkerLoader = new SafeTensorsLoader();
            talkerLoader.Load(talkerPath);
            SafeTensorsLoader codecLoader = new SafeTensorsLoader();
            codecLoader.Load(codecPath);

            Qwen3TtsPipeline pipeline = new Qwen3TtsPipeline(small ? Qwen3TtsConfig.Default_0_6B : Qwen3TtsConfig.Default_1_7B);
            // voice_clone (Base checkpoint only) also ships the ECAPA speaker encoder under speaker_encoder.*.
            pipeline.LoadWeights(talkerLoader.GetAllTensors(), talkerLoader.GetAllTensors(), codecLoader.GetAllTensors(),
                ecapa: mode == "voice_clone" ? talkerLoader.GetAllTensors() : null);
            Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: 512);
            Logs.Info($"[Audio][Qwen3-TTS] Loaded {repo} (mode={mode}, 12 Hz talker + MTP + codec, 24 kHz).");

            IDisposable?[] keep = [pipeline, talkerLoader, codecLoader];
            return new TtsRunner(pipeline.SampleRate, (backend, job) =>
            {
                // TTS needs the RAW byte-level token stream — the padded Encode (a diffusion text-encoder convention)
                // floods the talker's text stream and yields wrong-length garbage audio.
                int[] textTokens = [.. tokenizer.EncodeRaw(job.Text)];
                return mode switch
                {
                    "custom_voice" => pipeline.SynthesizeCustomVoice(backend, textTokens,
                        ResolveSpeakerToken(job.Voice), seed: job.Seed),
                    "voice_design" => pipeline.SynthesizeVoiceDesign(backend, textTokens, seed: job.Seed),
                    "voice_clone" => job.ReferenceMono24k is { Length: > 0 }
                        ? pipeline.SynthesizeVoiceClone(backend, textTokens, job.ReferenceMono24k, languageId: EnglishLanguageId, seed: job.Seed)
                        : throw new NotSupportedException("Qwen3-TTS voice_clone needs a reference voice clip — attach a ~3 s+ sample."),
                    _ => throw new InvalidOperationException($"Unknown Qwen3-TTS mode '{mode}'."),
                };
            }, keep);
        },
    };

    /// <summary>Maps a CustomVoice speaker NAME to its codec-space token. Only four ids are verified against the checkpoint (Ryan, Serena, Ono_Anna, Sohee); the other five documented voices have unverified placeholder ids in the config, so an unrecognized or unverified name falls back to the default rather than emitting a token that would synthesize the wrong voice.</summary>
    private static int ResolveSpeakerToken(string? speaker) => (speaker ?? string.Empty).Trim() switch
    {
        "Ryan" => 3_061,
        "Serena" => 3_066,
        "Ono_Anna" => 2_873,
        "Sohee" => 2_864,
        "" => EnglishSpeakerToken,
        _ => LogUnverified(speaker!),
    };

    private static int LogUnverified(string speaker)
    {
        Logs.Warning($"[Audio][Qwen3-TTS] Speaker '{speaker}' has no verified codec token yet — using the default voice. Verified: Ryan, Serena, Ono_Anna, Sohee.");
        return EnglishSpeakerToken;
    }

    /// <summary>Maps the variant hint (e.g. <c>1.7B-CustomVoice</c>) to the HuggingFace repo.</summary>
    private static string ResolveRepo(string variant)
    {
        string id = (variant ?? string.Empty).Trim();
        if (id.Contains('/', StringComparison.Ordinal))
        {
            return id;
        }
        if (id.Length == 0)
        {
            return DefaultRepo;
        }
        string size = id.Contains("0.6B", StringComparison.OrdinalIgnoreCase) ? "0.6B" : "1.7B";
        string flavor = id.Contains("CustomVoice", StringComparison.OrdinalIgnoreCase) ? "CustomVoice"
            : id.Contains("VoiceDesign", StringComparison.OrdinalIgnoreCase) ? "VoiceDesign"
            : "Base";
        return $"Qwen/Qwen3-TTS-12Hz-{size}-{flavor}";
    }

    private static string ResolveMode(string variant)
    {
        string id = (variant ?? string.Empty).Trim();
        if (id.Contains("CustomVoice", StringComparison.OrdinalIgnoreCase))
        {
            return "custom_voice";
        }
        if (id.Contains("VoiceDesign", StringComparison.OrdinalIgnoreCase))
        {
            return "voice_design";
        }
        return id.Length == 0 ? "custom_voice" : "voice_clone";
    }
}
