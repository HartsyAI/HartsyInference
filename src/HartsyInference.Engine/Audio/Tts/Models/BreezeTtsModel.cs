using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.BreezeTts;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>Breeze TTS 2 (<c>BreezeBlue/Breeze-TTS-2</c>) — an instruction-following TTS model: a T5Gemma2 text encoder
/// conditions a Qwen3 backbone that predicts the first of 16 codebooks per 12.5 Hz frame, a depth decoder fills in the
/// rest, and the bundled Qwen3-TTS-Tokenizer-12Hz vocoder renders 24 kHz audio. Three modes share one model:
/// <b>voice clone</b> (reference clip + its exact transcript), <b>voice design</b> (an <c>Instruction</c> describing the
/// voice, best with a guidance scale of about 4) and <b>voice direction</b> (reference + instruction). Inline events go
/// in the text as <c>(laugh)</c> (English) or <c>[笑]</c> (Chinese). The weights are research / non-commercial only.</summary>
internal static class BreezeTtsModel
{
    private const string Repo = "BreezeBlue/Breeze-TTS-2";
    private const string TokenizerFile = "tokenizer.json";
    private const string AudioTokenizerFolder = "audio_tokenizer";

    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = variant => (variant ?? string.Empty).Contains('/', StringComparison.Ordinal) ? variant! : Repo,
        ResolveFiles = async (_, cancel) =>
        {
            List<AudioModelFile> files = [new(TokenizerFile), new($"{AudioTokenizerFolder}/model.safetensors"), new($"{AudioTokenizerFolder}/config.json")];
            files.AddRange(await AudioCheckpoints.ResolveCheckpointFilesAsync(Repo, "tts", cancel).ConfigureAwait(false));
            return files;
        },
        LoadAsync = async (_, _, cancel) =>
        {
            string tokenizerPath = await AudioModelCache.GetAsync(Repo, TokenizerFile, category: "tts", ct: cancel).ConfigureAwait(false);
            (IReadOnlyDictionary<string, Tensor> audioTokenizer, IDisposable[] tokenizerLoaders) =
                await AudioCheckpoints.LoadSubfolderAsync(Repo, AudioTokenizerFolder, "tts", cancel).ConfigureAwait(false);
            (IReadOnlyDictionary<string, Tensor> model, IDisposable[] modelLoaders) =
                await AudioCheckpoints.LoadAsync(Repo, "tts", cancel).ConfigureAwait(false);

            using FileStream json = File.OpenRead(tokenizerPath);
            BreezeTts2Pipeline pipeline = new(BreezeTts2Config.Default, new SentencePieceBpeJson(json));
            pipeline.LoadWeights(model, audioTokenizer);
            Logs.Info("[Audio][Breeze] Loaded BreezeBlue/Breeze-TTS-2 (T5Gemma2 + Qwen3 backbone + depth decoder, 24 kHz).");
            IDisposable?[] keep = [pipeline, .. modelLoaders, .. tokenizerLoaders];
            return new TtsRunner(pipeline.SampleRate, (backend, job) => Synthesize(pipeline, backend, job), keep);
        },
    };

    private static float[] Synthesize(BreezeTts2Pipeline pipeline, IBackend backend, TtsJob job)
    {
        int[][]? referenceFrames = null;
        if (job.Reference is not null && job.Reference.Data.Length > 0)
        {
            if (string.IsNullOrWhiteSpace(job.RefText))
            {
                throw new ArgumentException(
                    "Breeze voice cloning needs the exact transcript of the reference clip (the request's reference text).");
            }
            referenceFrames = pipeline.EncodeReference(backend, AudioClipCodec.DecodeMono(job.Reference, pipeline.SampleRate));
        }
        return pipeline.Synthesize(backend, new BreezeTts2Pipeline.Request
        {
            Text = job.Text,
            Instruction = job.Instruction,
            ReferenceFrames = referenceFrames,
            ReferenceText = job.RefText,
            Speaker = job.SpeakerId is { } s ? $"S{s}" : "S0",
            CfgScale = job.CfgScale is { } c and > 0 ? (float)c : 1f,
            Temperature = job.Temperature is { } t ? (float)t : null,
            TopP = job.TopP is { } p ? (float)p : null,
            TopK = job.TopK,
            MaxFrames = job.MaxTokens,
            Seed = job.Seed,
            Cancel = job.Cancel,
        });
    }
}
