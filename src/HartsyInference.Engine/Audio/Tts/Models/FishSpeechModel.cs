using System.Runtime.CompilerServices;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.FishSpeech;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.PyTorch;

namespace HartsyInference.Engine.Audio;

/// <summary>Fish-Speech 1.5 (fishaudio/fish-speech-1.5) — a DualAR text2semantic model (Llama-style backbone plus a
/// depth transformer over 8 audio codebooks) decoded by the firefly-gan-vq codec to 44.1 kHz mono. The DualAR
/// weights, the codec, and the tiktoken vocab (plus its special-tokens sidecar) are separate repo files.
///
/// <para>Streaming: unlike Mimi's exact state-carrying reconstruction, this decodes each batch of newly-generated
/// frames independently through the (causal) Firefly codec — the same naive-but-real mechanism Fish-Speech's own
/// upstream inference engine uses (confirmed during research: <c>vq_manager.py</c>'s decode call is completely
/// stateless). See <see cref="FishSpeechPipeline.DecodeChunk"/> for the codec-level detail.</para></summary>
internal static class FishSpeechModel
{
    private const string Repo = "fishaudio/fish-speech-1.5";
    private const string ModelFile = "model.pth";
    private const string CodecFile = "firefly-gan-vq-fsq-8x1024-21hz-generator.pth";
    private const string TokenizerFile = "tokenizer.tiktoken";
    private const string SpecialTokensFile = "special_tokens.json";
    private const int FramesPerChunk = 25; // ~500ms @ Fish-Speech's ~50Hz frame rate (matches Kyutai's ~480ms convention).

    /// <summary>The Fish-Speech descriptor.</summary>
    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = variant => (variant ?? string.Empty).Contains('/', StringComparison.Ordinal) ? variant! : Repo,
        LoadAsync = async (_, _, cancel) =>
        {
            string modelPath = await AudioModelCache.GetAsync(Repo, ModelFile, category: "tts", ct: cancel).ConfigureAwait(false);
            string codecPath = await AudioModelCache.GetAsync(Repo, CodecFile, category: "tts", ct: cancel).ConfigureAwait(false);
            // FishSpeechTokenizer.Load auto-finds the special-tokens sidecar in the same cache dir, so fetch both.
            string tokenizerPath = await AudioModelCache.GetAsync(Repo, TokenizerFile, category: "tts", ct: cancel).ConfigureAwait(false);
            await AudioModelCache.GetAsync(Repo, SpecialTokensFile, category: "tts", ct: cancel).ConfigureAwait(false);

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

            Session session = new(pipeline, tokenizer);
            return new StreamingTtsRunner(pipeline.SampleRate, session.Synthesize, session.SynthesizeStream,
                pipeline, modelLoader, codecLoader);
        },
    };

    private sealed class Session(FishSpeechPipeline pipeline, FishSpeechTokenizer tokenizer)
    {
        /// <summary>Upstream v1.5 template (system turn + <|voice|> assistant opener). <|audio_start|> is NOT in
        /// the 1.5 vocab — it BPE-encodes as literal text and degrades generation.</summary>
        private static int[] BuildPrompt(FishSpeechTokenizer tokenizer, string text)
        {
            string prompt = $"{FishSpeechTokenizer.ImStart}system\nSpeak out the provided text.{FishSpeechTokenizer.ImEnd}"
                + $"{FishSpeechTokenizer.ImStart}user\n{text}{FishSpeechTokenizer.ImEnd}"
                + $"{FishSpeechTokenizer.ImStart}assistant\n{FishSpeechTokenizer.Voice}";
            return tokenizer.Encode(prompt);
        }

        public float[] Synthesize(IBackend backend, TtsJob job)
        {
            int[] tokens = BuildPrompt(tokenizer, job.Text);
            // Upstream's inference default for max_new_tokens is 0 (= run to the stop token).
            return pipeline.Synthesize(backend, tokens, endToken: tokenizer.ImEndId,
                maxFrames: job.MaxTokens is > 0 ? job.MaxTokens.Value : 0, seed: job.Seed,
                normalizeLoudness: job.NormalizeLoudness ?? true);
        }

        public async IAsyncEnumerable<AudioChunk> SynthesizeStream(IBackend backend, TtsJob job, [EnumeratorCancellation] CancellationToken cancel)
        {
            int[] tokens = BuildPrompt(tokenizer, job.Text);
            int maxFrames = job.MaxTokens is > 0 ? job.MaxTokens.Value : 0;
            using AudioStreamer streamer = new();
            long samplesEmitted = 0;

            Task producer = Task.Run(() =>
            {
                List<int[]> pending = new(FramesPerChunk);
                try
                {
                    pipeline.Synthesize(backend, tokens, endToken: tokenizer.ImEndId, maxFrames: maxFrames,
                        seed: job.Seed, normalizeLoudness: false, onFrame: frame =>
                        {
                            pending.Add(frame);
                            if (pending.Count < FramesPerChunk) return;
                            EmitChunk(backend, streamer, pending, cancel, ref samplesEmitted);
                            pending.Clear();
                        });
                    if (pending.Count > 0)
                    {
                        EmitChunk(backend, streamer, pending, cancel, ref samplesEmitted);
                    }
                }
                finally
                {
                    streamer.Complete();
                }
            }, cancel);

            try
            {
                await foreach (AudioChunk chunk in streamer.ReadAllAsync(cancel).ConfigureAwait(false))
                {
                    yield return chunk;
                }
            }
            finally
            {
                await producer.ConfigureAwait(false);
            }
        }

        private void EmitChunk(IBackend backend, AudioStreamer streamer, List<int[]> pending, CancellationToken cancel, ref long samplesEmitted)
        {
            float[] pcm = pipeline.DecodeChunk(backend, pending);
            AudioChunk chunk = new(pcm, pipeline.SampleRate, Channels: 1, samplesEmitted);
            samplesEmitted += pcm.Length;
            streamer.Put(chunk, cancel).AsTask().GetAwaiter().GetResult();
        }
    }
}
