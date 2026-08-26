using System.Runtime.CompilerServices;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Models.Bark;
using HartsyInference.Audio.Models.Codecs.EnCodec;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>Bark (suno/bark) — 3-stage GPT cascade (semantic → coarse → fine) + EnCodec 24 kHz, text via the
/// multilingual BERT WordPiece tokenizer plus Bark's text-encoding offset.
///
/// <para>Streaming here is narrower than Kyutai/CSM/Orpheus: the fine stage is a non-autoregressive, full-context
/// refinement over the whole coarse output, so no audio can start until all three stages finish — see
/// <see cref="BarkPipeline.SynthesizeStreamChunks"/> for what streaming actually buys here (chunked EnCodec
/// decode after generation, not incremental emission during it) and why (EnCodec's LSTM bottleneck is stateless
/// in every known reference implementation, confirmed during research, not assumed).</para></summary>
internal static class BarkTtsModel
{
    private const string Repo = "suno/bark";

    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => Repo,
        ResolveFiles = async (_, cancel) =>
        {
            // Bark's text frontend borrows BERT's vocab from a third-party repo.
            List<AudioModelFile> files = [new AudioModelFile("vocab.txt", Repo: "google-bert/bert-base-multilingual-cased")];
            // Mirrors LoadBarkWeightsAsync's safetensors-then-pickle preference, probed rather than downloaded.
            bool safetensors = await AudioModelCache.ExistsAsync(Repo, "model.safetensors", "tts", ct: cancel).ConfigureAwait(false);
            files.Add(new AudioModelFile(safetensors ? "model.safetensors" : "pytorch_model.bin"));
            return files;
        },
        LoadAsync = async (_, _, cancel) =>
        {
            string vocabPath = await AudioModelCache.GetAsync("google-bert/bert-base-multilingual-cased", "vocab.txt", category: "tts", ct: cancel).ConfigureAwait(false);
            (IReadOnlyDictionary<string, Tensor> dict, IDisposable loader) = await LoadBarkWeightsAsync(cancel).ConfigureAwait(false);

            BarkConfig config = BarkConfig.Full;
            BarkCausalStage semantic = new BarkCausalStage(config.Stage, config.SemanticInputVocab, config.SemanticOutputVocab);
            semantic.LoadWeights(dict, "semantic");
            BarkCausalStage coarse = new BarkCausalStage(config.Stage, config.CoarseVocab, config.CoarseVocab);
            coarse.LoadWeights(dict, "coarse_acoustics");
            BarkFineModel fine = new BarkFineModel(config);
            fine.LoadWeights(dict, "fine_acoustics");

            Dictionary<string, Tensor> codecRaw = new Dictionary<string, Tensor>(StringComparer.Ordinal);
            foreach (KeyValuePair<string, Tensor> entry in dict)
            {
                if (entry.Key.StartsWith("codec_model.", StringComparison.Ordinal))
                {
                    codecRaw[entry.Key["codec_model.".Length..]] = entry.Value;
                }
            }
            EnCodec encodec = new EnCodec(EnCodecConfig.EnCodec24kHz);
            encodec.LoadWeights(MusicGenCheckpointConverter.ConvertEnCodec(codecRaw));

            BarkPipeline pipeline = new BarkPipeline(config, semantic, coarse, fine, encodec);
            BertWordPieceTokenizer bert = new BertWordPieceTokenizer(vocabPath, lowerCase: false);
            Logs.Info("[Audio][Bark] Loaded suno/bark (3-stage GPT + EnCodec 24 kHz).");

            // Loader kept alive: the F32 stage weights reference its tensors.
            Session session = new(pipeline, config, bert);
            return new StreamingTtsRunner(config.SampleRate, session.Synthesize, session.SynthesizeStream, pipeline, loader);
        },
    };

    /// <summary>Loads the Bark transformers checkpoint — safetensors preferred, pickle fallback.</summary>
    private static async Task<(IReadOnlyDictionary<string, Tensor> Dict, IDisposable Loader)> LoadBarkWeightsAsync(CancellationToken cancel)
    {
        try
        {
            string path = await AudioModelCache.GetAsync(Repo, "model.safetensors", category: "tts", ct: cancel).ConfigureAwait(false);
            SafeTensorsLoader loader = new SafeTensorsLoader();
            loader.Load(path);
            return (loader.GetAllTensors(), loader);
        }
        catch (FileNotFoundException ex)
        {
            Logs.Debug($"[Audio][Bark] No model.safetensors ({ex.Message}); loading pytorch_model.bin.");
            string path = await AudioModelCache.GetAsync(Repo, "pytorch_model.bin", category: "tts", ct: cancel).ConfigureAwait(false);
            PytorchPickleLoader loader = new PytorchPickleLoader();
            loader.Load(path);
            return (loader.GetAllTensors(), loader);
        }
    }

    private sealed class Session(BarkPipeline pipeline, BarkConfig config, BertWordPieceTokenizer bert)
    {
        public float[] Synthesize(IBackend backend, TtsJob job)
            => pipeline.Synthesize(backend, AudioTextFrontend.BarkText(bert, job.Text, config.TextEncodingOffset),
                job.Seed, 768, job.Temperature, job.WaveformTemperature);

        public async IAsyncEnumerable<AudioChunk> SynthesizeStream(IBackend backend, TtsJob job, [EnumeratorCancellation] CancellationToken cancel)
        {
            int[] textTokenIds = AudioTextFrontend.BarkText(bert, job.Text, config.TextEncodingOffset);
            using AudioStreamer streamer = new();
            long samplesEmitted = 0;

            Task producer = Task.Run(() =>
            {
                try
                {
                    foreach (float[] chunk in pipeline.SynthesizeStreamChunks(backend, textTokenIds, job.Seed, 768, job.Temperature, job.WaveformTemperature))
                    {
                        cancel.ThrowIfCancellationRequested();
                        AudioChunk audioChunk = new(chunk, config.SampleRate, Channels: 1, samplesEmitted);
                        samplesEmitted += chunk.Length;
                        streamer.Put(audioChunk, cancel).AsTask().GetAwaiter().GetResult();
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
    }
}
