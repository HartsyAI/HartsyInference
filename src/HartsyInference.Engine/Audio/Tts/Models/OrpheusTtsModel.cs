using System.Runtime.CompilerServices;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Models.Orpheus;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Engine.Audio;

/// <summary>Orpheus TTS — Llama-3.2-3B LM + SNAC 24 kHz, from a non-gated mirror of the license-gated release.
///
/// <para>SNAC's decoder uses symmetric ("same") padding throughout (verified against the real weights — no
/// causal-conv variant exists for this architecture, unlike Mimi), so its streaming path can't be an exact
/// state-carrying reconstruction. It instead follows the official Orpheus reference client's own recipe
/// (<c>orpheus_tts/decoder.py</c>): decode a 4-group sliding window and keep only the middle group's span,
/// discarding the window's own edges — see <see cref="OrpheusPipeline.SynthesizeStreamChunks"/> for the full
/// design (including the two things that recipe gets wrong that this port doesn't: dropped head/tail audio, and
/// a periodic noise artifact from windowed decode reusing a fixed-seed RNG).</para></summary>
internal static class OrpheusTtsModel
{
    private const string BackboneRepo = "unsloth/orpheus-3b-0.1-ft";
    private const string SnacRepo = "hubertsiuzdak/snac_24khz";

    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => BackboneRepo,
        ResolveFiles = async (_, cancel) =>
        {
            // SNAC codec first, backbone last: the backbone is the artifact that marks Orpheus installed.
            List<AudioModelFile> files = [];
            foreach (AudioModelFile snac in await AudioCheckpoints.ResolveCheckpointFilesAsync(SnacRepo, "tts", cancel).ConfigureAwait(false))
            {
                files.Add(snac with { Repo = SnacRepo });
            }
            files.AddRange(await AudioCheckpoints.ResolveCheckpointFilesAsync(BackboneRepo, "tts", cancel).ConfigureAwait(false));
            return files;
        },
        LoadAsync = async (_, _, cancel) =>
        {
            (IReadOnlyDictionary<string, Tensor> backbone, IDisposable[] backboneLoaders) =
                await AudioCheckpoints.LoadAsync(BackboneRepo, "tts", cancel).ConfigureAwait(false);
            (IReadOnlyDictionary<string, Tensor> snac, IDisposable[] snacLoaders) =
                await AudioCheckpoints.LoadAsync(SnacRepo, "tts", cancel).ConfigureAwait(false);
            OrpheusPipeline pipeline = new OrpheusPipeline(OrpheusConfig.Orpheus3B);
            pipeline.LoadWeights(backbone, snac);
            Logs.Info("[Audio][Orpheus] Loaded unsloth/orpheus-3b-0.1-ft (Llama-3.2-3B + SNAC 24 kHz).");
            IDisposable?[] keep = [pipeline, .. backboneLoaders, .. snacLoaders];
            Session session = new(pipeline);
            return new StreamingTtsRunner(pipeline.SampleRate, session.Synthesize, session.SynthesizeStream, keep);
        },
    };

    private sealed class Session(OrpheusPipeline pipeline)
    {
        public float[] Synthesize(IBackend backend, TtsJob job)
            => pipeline.Synthesize(backend, AudioTextFrontend.OrpheusText(job.Text), seed: job.Seed);

        /// <summary>Runs the windowed-decode generator (<see cref="OrpheusPipeline.SynthesizeStreamChunks"/>) on a background thread and pushes each already-decoded PCM chunk through an <see cref="AudioStreamer"/> — simpler than Kyutai/CSM's <c>StreamingCodecDecoder</c> wiring because the windowing/redistribution/ decode work all happens inside the pipeline itself; this layer only bridges sync-iterator to async-stream.</summary>
        public async IAsyncEnumerable<AudioChunk> SynthesizeStream(IBackend backend, TtsJob job, [EnumeratorCancellation] CancellationToken cancel)
        {
            int[] textTokenIds = AudioTextFrontend.OrpheusText(job.Text);
            using AudioStreamer streamer = new();
            long samplesEmitted = 0;

            Task producer = Task.Run(() =>
            {
                try
                {
                    foreach (float[] chunk in pipeline.SynthesizeStreamChunks(backend, textTokenIds, seed: job.Seed))
                    {
                        cancel.ThrowIfCancellationRequested();
                        AudioChunk audioChunk = new(chunk, pipeline.SampleRate, Channels: 1, samplesEmitted);
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
