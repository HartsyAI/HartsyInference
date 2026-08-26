using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Codecs.NeuCodec;
using HartsyInference.Audio.Models.NeuTts;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Audio.Phonemizer;
using HartsyInference.Audio.Phonemizer.Espeak;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>NeuTTS Air (neuphonic/neutts-air) — a Qwen2.5-0.5B LM that emits a single NeuCodec FSQ stream, decoded to 24 kHz. Text is espeak-phonemized to IPA and framed in the upstream chat template; when a reference clip is supplied it is encoded to FSQ codes that prime generation and its transcript's phones are prepended.</summary>
internal static class NeuTtsModel
{
    private const string BackboneRepo = "neuphonic/neutts-air";
    private const string CodecRepo = "neuphonic/neucodec";
    private const string EspeakLanguage = "en-us";   // upstream BACKBONE_LANGUAGE_MAP["neuphonic/neutts-air"]

    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = variant => (variant ?? string.Empty).Contains('/', StringComparison.Ordinal) ? variant! : BackboneRepo,
        ResolveFiles = async (_, cancel) =>
        {
            // Codec first; the backbone is what marks NeuTTS installed.
            List<AudioModelFile> files = [];
            foreach (AudioModelFile codecFile in await AudioCheckpoints.ResolveCheckpointFilesAsync(CodecRepo, "tts", cancel).ConfigureAwait(false))
            {
                files.Add(codecFile with { Repo = CodecRepo });
            }
            files.AddRange(await AudioCheckpoints.ResolveCheckpointFilesAsync(BackboneRepo, "tts", cancel).ConfigureAwait(false));
            return files;
        },
        LoadAsync = async (_, _, cancel) =>
        {
            (IReadOnlyDictionary<string, Tensor> backbone, IDisposable[] backboneLoaders) =
                await AudioCheckpoints.LoadAsync(BackboneRepo, "tts", cancel).ConfigureAwait(false);
            (IReadOnlyDictionary<string, Tensor> codec, IDisposable[] codecLoaders) =
                await AudioCheckpoints.LoadAsync(CodecRepo, "tts", cancel).ConfigureAwait(false);

            NeuTtsConfig config = NeuTtsConfig.Air;
            NeuTtsPipeline pipeline = new NeuTtsPipeline(config);
            pipeline.LoadWeights(backbone, codec);
            NeuCodecEncoder? encoder = null;
            if (codec.ContainsKey("acoustic_encoder.conv1.weight"))
            {
                encoder = new NeuCodecEncoder(NeuCodecEncoderConfig.Default);
                encoder.LoadWeights(codec);
            }
            // Upstream phonemizes with espeak-ng (en-us, with_stress, preserve_punctuation); raw text is
            // out-of-distribution and produces garble.
            IPhonemizer phonemizer = EspeakPhonemizer.FromCache(EspeakLanguage);
            Qwen2Tokenizer tokenizer = new Qwen2Tokenizer();
            Logs.Info($"[Audio][NeuTTS] Loaded neuphonic/neutts-air (Qwen2.5-0.5B + NeuCodec, 24 kHz; cloning "
                + $"{(encoder is null ? "unavailable — X-Codec2 encoder port pending" : "when a reference is supplied")}).");

            // Mirrors upstream _to_phones: phonemize, then whitespace-normalize.
            string Phones(string text) =>
                string.Join(' ', phonemizer.PhonemizeToIpa(text, EspeakLanguage).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));

            IDisposable?[] keep = encoder is null ? [pipeline, tokenizer, .. backboneLoaders, .. codecLoaders]
                : [pipeline, encoder, tokenizer, .. backboneLoaders, .. codecLoaders];
            return new TtsRunner(pipeline.SampleRate, (backend, job) =>
            {
                int[] referenceCodes = [];
                string phones = Phones(job.Text);
                if (job.Reference is not null && job.Reference.Data.Length > 0)
                {
                    if (encoder is null)
                    {
                        throw new NotSupportedException(
                            "NeuTTS reference-voice cloning needs the X-Codec2 encoder, which this checkpoint does not "
                            + "ship. Clear the voice reference to use the default voice.");
                    }
                    float[] reference16k = AudioClipCodec.DecodeMono(job.Reference, 16_000);
                    if (reference16k.Length > 0)
                    {
                        referenceCodes = encoder.Encode(backend, reference16k);
                    }
                    if (!string.IsNullOrWhiteSpace(job.RefText))
                    {
                        phones = $"{Phones(job.RefText)} {phones}";   // upstream joins ref + target phones with " "
                    }
                }
                int[] promptPrefix = NeuTtsPromptBuilder.BuildPromptPrefix(config, tokenizer.EncodeRawByteLevel, phones);
                return pipeline.Synthesize(backend, promptPrefix, referenceCodes, seed: job.Seed);
            }, keep);
        },
    };
}
