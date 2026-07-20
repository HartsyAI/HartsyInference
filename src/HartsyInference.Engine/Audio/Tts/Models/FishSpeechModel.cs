using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.FishSpeech;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.PyTorch;

namespace HartsyInference.Engine.Audio;

/// <summary>Fish-Speech 1.5 (fishaudio/fish-speech-1.5) — a DualAR text2semantic model (Llama-style backbone plus a
/// depth transformer over 8 audio codebooks) decoded by the firefly-gan-vq codec to 44.1 kHz mono. The DualAR
/// weights, the codec, and the tiktoken vocab (plus its special-tokens sidecar) are separate repo files.</summary>
internal static class FishSpeechModel
{
    private const string Repo = "fishaudio/fish-speech-1.5";
    private const string ModelFile = "model.pth";
    private const string CodecFile = "firefly-gan-vq-fsq-8x1024-21hz-generator.pth";
    private const string TokenizerFile = "tokenizer.tiktoken";
    private const string SpecialTokensFile = "special_tokens.json";

    /// <summary>The Fish-Speech descriptor.</summary>
    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = variant => (variant ?? string.Empty).Contains('/', StringComparison.Ordinal) ? variant! : Repo,
        LoadAsync = async (_, cancel) =>
        {
            string modelPath = await AudioModelCache.GetAsync(Repo, ModelFile, ct: cancel).ConfigureAwait(false);
            string codecPath = await AudioModelCache.GetAsync(Repo, CodecFile, ct: cancel).ConfigureAwait(false);
            // FishSpeechTokenizer.Load auto-finds the special-tokens sidecar in the same cache dir, so fetch both.
            string tokenizerPath = await AudioModelCache.GetAsync(Repo, TokenizerFile, ct: cancel).ConfigureAwait(false);
            await AudioModelCache.GetAsync(Repo, SpecialTokensFile, ct: cancel).ConfigureAwait(false);

            PytorchPickleLoader modelLoader = new PytorchPickleLoader();
            modelLoader.Load(modelPath);
            PytorchPickleLoader codecLoader = new PytorchPickleLoader();
            codecLoader.Load(codecPath);

            FishSpeechTokenizer tokenizer = new FishSpeechTokenizer();
            tokenizer.Load(tokenizerPath);
            if (tokenizer.ImEndId < 0)
            {
                throw new InvalidOperationException(
                    $"Fish-Speech tokenizer at '{tokenizerPath}' is missing the '<|im_end|>' stop token — the DualAR "
                    + "pipeline cannot determine when to stop. Verify the tokenizer asset.");
            }

            FishSpeechPipeline pipeline = new FishSpeechPipeline(FishSpeechConfig.V1_5);
            pipeline.LoadWeights(modelLoader.GetAllTensors(), codecLoader.GetAllTensors());
            Logs.Info("[Audio][FishSpeech] Loaded fishaudio/fish-speech-1.5 (DualAR + firefly-gan-vq, 44.1 kHz).");

            return new TtsRunner(pipeline.SampleRate, (backend, job) =>
            {
                // Upstream v1.5 template (system turn + <|voice|> assistant opener). <|audio_start|> is NOT in the 1.5
                // vocab — it BPE-encodes as literal text and degrades generation.
                string prompt = $"{FishSpeechTokenizer.ImStart}system\nSpeak out the provided text.{FishSpeechTokenizer.ImEnd}"
                    + $"{FishSpeechTokenizer.ImStart}user\n{job.Text}{FishSpeechTokenizer.ImEnd}"
                    + $"{FishSpeechTokenizer.ImStart}assistant\n{FishSpeechTokenizer.Voice}";
                int[] tokens = tokenizer.Encode(prompt);
                return pipeline.Synthesize(backend, tokens, endToken: tokenizer.ImEndId, seed: job.Seed);
            }, pipeline, modelLoader, codecLoader);
        },
    };
}
