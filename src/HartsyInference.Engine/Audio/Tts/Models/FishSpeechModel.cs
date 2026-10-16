using System.Runtime.CompilerServices;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.FishSpeech;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>Fish-Speech: a DualAR text2semantic model (Llama-style backbone plus a depth transformer over 8 audio codebooks)
/// decoded by the firefly-gan-vq codec to 44.1 kHz mono. One descriptor serves both checkpoints; the variant picks the repo and
/// its layout (see <see cref="Variant"/>). The default is fish-speech-1.5 (tiktoken vocab). <c>fish-speech-1.4</c> loads the 1.4
/// layout: HF tokenizer.json, the single &lt;|semantic|&gt; sentinel, one-based codebook values and the plain chat prompt.
///
/// <para>Streaming: each batch of newly generated frames is decoded independently through the (causal) Firefly codec, the same
/// stateless mechanism upstream's inference engine uses (<c>vq_manager.py</c>). See <see cref="FishSpeechPipeline.DecodeChunk"/>.</para></summary>
internal static class FishSpeechModel
{
    private const int FramesPerChunk = 25; // ~500ms @ Fish-Speech's ~50Hz frame rate (matches Kyutai's ~480ms convention).
    private const string ModelFile = "model.pth";
    private const string CodecFile = "firefly-gan-vq-fsq-8x1024-21hz-generator.pth";

    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = variant => Variant.Parse(variant).Repo,
        ResolveFiles = (variant, _) => Task.FromResult<IReadOnlyList<AudioModelFile>>(Variant.Parse(variant).Files()),
        // Neither layout clones yet (the Firefly encoder is not ported). Reject a reference here, before the download and load,
        // instead of ignoring it after them.
        ValidateReferenceSeconds = _ => throw new NotSupportedException(
            "Fish-Speech voice cloning is not implemented yet (it needs the Firefly encoder). Drop the reference clip, "
            + "or use a cloning model."),
        LoadAsync = (_, variant, cancel) => LoadAsync(Variant.Parse(variant), cancel),
    };

    /// <summary>The checkpoint a variant names. The 1.4 layout is picked by its repo; any other repo keeps the 1.5 layout.</summary>
    internal readonly record struct Variant(string Repo, bool Layout14)
    {
        private const string Repo15 = "fishaudio/fish-speech-1.5";
        private const string Repo14 = "fishaudio/fish-speech-1.4";

        /// <summary>Accepts the bare family id, <c>fish-speech-1.5</c> or <c>1.5</c> (the default), <c>fish-speech-1.4</c> or
        /// <c>1.4</c>, or an org/repo. Anything else throws, so a request for a checkpoint this engine does not run never loads a
        /// different one.</summary>
        internal static Variant Parse(string? variant)
        {
            string token = (variant ?? string.Empty).Trim();
            if (token.Contains('/', StringComparison.Ordinal))
            {
                return new Variant(token, string.Equals(token, Repo14, StringComparison.OrdinalIgnoreCase));
            }
            return token.ToLowerInvariant() switch
            {
                "" or "fishspeech" or "fish-speech-1.5" or "1.5" => new Variant(Repo15, false),
                "fish-speech-1.4" or "1.4" => new Variant(Repo14, true),
                _ => throw new ArgumentException(
                    $"Unknown Fish-Speech variant '{variant}'. Use fish-speech-1.5 (the default), fish-speech-1.4, "
                    + "or an org/repo in the 1.5 layout."),
            };
        }

        /// <summary>The files this layout pulls: companions first and the weights last, since the weights mark the model installed.</summary>
        internal IReadOnlyList<AudioModelFile> Files() => Layout14
            ? [new("tokenizer.json"), new(CodecFile), new(ModelFile)]
            : [new("tokenizer.tiktoken"), new("special_tokens.json"), new(CodecFile), new(ModelFile)];
    }

    private static async Task<ITtsRunner> LoadAsync(Variant variant, CancellationToken cancel)
    {
        Dictionary<string, string> paths = [];
        foreach (AudioModelFile file in variant.Files())
        {
            paths[file.Name] = await AudioModelCache.GetAsync(variant.Repo, file.Name, category: "tts", ct: cancel).ConfigureAwait(false);
        }

        AnyFormatCheckpointLoader modelLoader = new AnyFormatCheckpointLoader();
        modelLoader.Load(paths[ModelFile]);
        AnyFormatCheckpointLoader codecLoader = new AnyFormatCheckpointLoader();
        codecLoader.Load(paths[CodecFile]);

        return variant.Layout14
            ? Build14(variant, paths["tokenizer.json"], modelLoader, codecLoader)
            : Build15(variant, paths["tokenizer.tiktoken"], modelLoader, codecLoader);
    }

    private static ITtsRunner Build15(Variant variant, string tokenizerPath, AnyFormatCheckpointLoader modelLoader,
        AnyFormatCheckpointLoader codecLoader)
    {
        // FishSpeechTokenizer.Load auto-finds the special-tokens sidecar in the same cache dir.
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
        Logs.Info($"[Audio][FishSpeech] Loaded {variant.Repo} (DualAR + firefly-gan-vq, 44.1 kHz).");

        Session session = new(pipeline, text => Session.BuildPrompt(tokenizer, text), tokenizer.ImEndId);
        return new StreamingTtsRunner(pipeline.SampleRate, session.Synthesize, session.SynthesizeStream,
            pipeline, modelLoader, codecLoader);
    }

    private static ITtsRunner Build14(Variant variant, string tokenizerPath, AnyFormatCheckpointLoader modelLoader,
        AnyFormatCheckpointLoader codecLoader)
    {
        GgufTokenizer tokenizer;
        using (FileStream json = File.OpenRead(tokenizerPath)) tokenizer = HfTokenizerJson.LoadByteLevelBpe(json);
        int? imEnd = tokenizer.SpecialId(FishSpeechTokenizer.ImEnd);
        int? semantic = tokenizer.SpecialId(FishSpeechTokenizer.PlainSemantic);
        if (imEnd is null || semantic != FishSpeechConfig.V1_4.SemanticTokenId)
        {
            throw new InvalidOperationException(
                $"Fish-Speech 1.4 tokenizer at '{tokenizerPath}' does not carry '<|im_end|>' and the '<|semantic|>' id "
                + $"{FishSpeechConfig.V1_4.SemanticTokenId}. Verify the tokenizer asset.");
        }

        FishSpeechPipeline pipeline = new FishSpeechPipeline(FishSpeechConfig.V1_4);
        pipeline.LoadWeights(modelLoader.GetAllTensors(), codecLoader.GetAllTensors());
        Logs.Info($"[Audio][FishSpeech] Loaded {variant.Repo} (DualAR + firefly-gan-vq, 44.1 kHz, 1.4 layout).");

        // addSpecial matches the <|...|> literals (HF always matches added tokens); it adds no BOS or EOS.
        Session session = new(pipeline, text => tokenizer.Encode(FishSpeechPrompt.Plain14(text), addSpecial: true), imEnd!.Value);
        return new StreamingTtsRunner(pipeline.SampleRate, session.Synthesize, session.SynthesizeStream,
            pipeline, modelLoader, codecLoader);
    }

    private sealed class Session(FishSpeechPipeline pipeline, Func<string, int[]> encodePrompt, int endToken)
    {
        /// <summary>Upstream v1.5 template (system turn + <|voice|> assistant opener). <|audio_start|> is NOT in the 1.5 vocab — it BPE-encodes as literal text and degrades generation.</summary>
        internal static int[] BuildPrompt(FishSpeechTokenizer tokenizer, string text)
        {
            string prompt = $"{FishSpeechTokenizer.ImStart}system\nSpeak out the provided text.{FishSpeechTokenizer.ImEnd}"
                + $"{FishSpeechTokenizer.ImStart}user\n{text}{FishSpeechTokenizer.ImEnd}"
                + $"{FishSpeechTokenizer.ImStart}assistant\n{FishSpeechTokenizer.Voice}";
            return tokenizer.Encode(prompt);
        }

        public float[] Synthesize(IBackend backend, TtsJob job)
        {
            int[] tokens = encodePrompt(job.Text);
            // Upstream's inference default for max_new_tokens is 0 (= run to the stop token).
            return pipeline.Synthesize(backend, tokens, endToken: endToken,
                maxFrames: job.MaxTokens is > 0 ? job.MaxTokens.Value : 0, seed: job.Seed,
                normalizeLoudness: job.NormalizeLoudness ?? true);
        }

        public async IAsyncEnumerable<AudioChunk> SynthesizeStream(IBackend backend, TtsJob job, [EnumeratorCancellation] CancellationToken cancel)
        {
            int[] tokens = encodePrompt(job.Text);
            int maxFrames = job.MaxTokens is > 0 ? job.MaxTokens.Value : 0;
            using AudioStreamer streamer = new();
            long samplesEmitted = 0;

            Task producer = Task.Run(() =>
            {
                List<int[]> pending = new(FramesPerChunk);
                try
                {
                    pipeline.Synthesize(backend, tokens, endToken: endToken, maxFrames: maxFrames,
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
