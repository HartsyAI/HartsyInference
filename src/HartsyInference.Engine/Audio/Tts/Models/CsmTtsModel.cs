using System.Runtime.CompilerServices;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Audio.Models.Codecs.Mimi;
using HartsyInference.Audio.Models.Csm;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;

namespace HartsyInference.Engine.Audio;

/// <summary>Sesame CSM-1B — dual-transformer conversational TTS + Mimi 24 kHz, from a non-gated mirror of the HF
/// <c>transformers</c> re-export (<c>unsloth/csm-1b</c>). Its keys are the engine's standard Llama layout under
/// <c>backbone_model.</c>/<c>depth_decoder.model.</c> prefixes, plus two combined tensors (the audio embed table
/// and the stacked codebook heads) that <see cref="CsmWeightRemap"/> splits per-codebook. The checkpoint also
/// bundles its own 32-codebook Mimi under a <c>codec_model.</c> prefix — used instead of the separately-published
/// <c>kyutai/mimi</c> (8 codebooks: a mismatch against CSM's 32-codebook depth decoder). (A prior mirror choice,
/// <c>nielsr/csm-1b</c>, ships the original torchtune-style keys — <c>attn.q_proj</c>, <c>sa_norm.scale</c> —
/// which don't match this loader at all; see <see cref="CsmWeightRemap"/>.)
///
/// <para>Decodes through the SAME <see cref="Mimi"/> codec class Kyutai TTS uses. Streaming reuses
/// <see cref="Mimi.DecodeStreaming"/>/<see cref="MimiDecoderStreamState"/> unchanged from the Kyutai work — the
/// codec-level correctness (chunk boundaries reconstruct monolithic decode exactly) is already proven by
/// <c>MimiStreamParityTests</c>; this file only adds the frame-batching/producer-consumer plumbing on top, mirroring
/// <see cref="KyutaiTtsModel"/>'s <c>SynthesizeStream</c> shape. CSM's AR loop is simpler than Kyutai's Moshi
/// backbone: every step yields one immediately-valid frame (no delayed-streams warmup/skip bookkeeping), so
/// <see cref="CsmPipeline.Synthesize"/>'s <c>onFrame</c> callback fires for every real frame with nothing to
/// filter.</para></summary>
internal static class CsmTtsModel
{
    private const string Repo = "unsloth/csm-1b";

    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => Repo,
        LoadAsync = async (_, _, cancel) =>
        {
            (IReadOnlyDictionary<string, Tensor> modelDict, IDisposable[] modelLoaders) =
                await AudioCheckpoints.LoadAsync(Repo, "tts", cancel).ConfigureAwait(false);
            CsmModel model = new CsmModel(CsmConfig.V1B);
            model.LoadWeights(CsmWeightRemap.Remap(modelDict));
            Mimi mimi = new Mimi(MimiConfig.Mimi24kHzDsm);
            mimi.LoadWeights(CsmWeightRemap.ExtractMimiWeights(modelDict));
            CsmPipeline pipeline = new CsmPipeline(CsmConfig.V1B, model, mimi);
            Logs.Info("[Audio][CSM] Loaded unsloth/csm-1b (dual-transformer + bundled Mimi, 32 codebooks, 24 kHz).");
            IDisposable?[] keep = [pipeline, .. modelLoaders];
            Session session = new(pipeline, mimi);
            return new StreamingTtsRunner(24_000, session.Synthesize, session.SynthesizeStream, keep);
        },
    };

    /// <summary>Owns the loaded pipeline; both entry points share it. Disposal is owned by the <see cref="StreamingTtsRunner"/>'s own <c>keep</c> array in <see cref="Descriptor"/>, not by this class.</summary>
    private sealed class Session(CsmPipeline pipeline, Mimi mimi)
    {
        public float[] Synthesize(IBackend backend, TtsJob job)
            => pipeline.Synthesize(backend, AudioTextFrontend.CsmText(job.Text, job.SpeakerId ?? 0), seed: job.Seed);

        /// <summary>Streaming counterpart: runs the AR loop on a background thread, batching every <c>FramesPerChunk</c> genuine frames into a <see cref="Mimi.DecodeStreaming"/> call against one shared <see cref="MimiDecoderStreamState"/> for the whole utterance — identical shape to <c>KyutaiTtsModel.SynthesizeStream</c>, including the producer/consumer exception-propagation discipline (no inline catch; the decoder's <c>Complete()</c> runs in a <c>finally</c> so a faulted/cancelled producer always unblocks the consumer, and the outer <c>finally</c> always awaits the producer task so its exception surfaces even if the caller stops enumerating early).</summary>
        public async IAsyncEnumerable<HartsyInference.Audio.Streaming.AudioChunk> SynthesizeStream(
            IBackend backend, TtsJob job, [EnumeratorCancellation] CancellationToken cancel)
        {
            const int FramesPerChunk = 6; // 80ms/frame * 6 = 480ms, matching Kyutai's chunk granularity.
            int[] textTokenIds = AudioTextFrontend.CsmText(job.Text, job.SpeakerId ?? 0);
            using MimiDecoderStreamState decodeState = new();
            using StreamingCodecDecoder<Mimi> decoder = new(mimi, backend,
                (codec, be, codes, batch, tFrames) => codec.DecodeStreaming(be, codes, batch, tFrames, decodeState), 24_000);

            Task producer = Task.Run(() =>
            {
                List<int[]> pending = new(FramesPerChunk);
                try
                {
                    pipeline.Synthesize(backend, textTokenIds, seed: job.Seed, onFrame: frame =>
                    {
                        pending.Add(frame);
                        if (pending.Count < FramesPerChunk)
                        {
                            return;
                        }
                        SubmitPending(decoder, pending, cancel);
                        pending.Clear();
                    });
                    if (pending.Count > 0)
                    {
                        SubmitPending(decoder, pending, cancel);
                    }
                }
                finally
                {
                    decoder.Complete();
                }
            }, cancel);

            try
            {
                await foreach (HartsyInference.Audio.Streaming.AudioChunk chunk in decoder.ReadAllAsync(cancel).ConfigureAwait(false))
                {
                    yield return chunk;
                }
            }
            finally
            {
                await producer.ConfigureAwait(false);
            }
        }

        private static unsafe void SubmitPending(StreamingCodecDecoder<Mimi> decoder, List<int[]> frames, CancellationToken cancel)
        {
            int numCodebooks = frames[0].Length;
            int n = frames.Count;
            Tensor codeTensor = new(new TensorShape(1, numCodebooks, n), DType.I32);
            int* codePtr = (int*)codeTensor.DataPointer;
            for (int cb = 0; cb < numCodebooks; cb++)
            {
                for (int f = 0; f < n; f++)
                {
                    codePtr[(long)cb * n + f] = frames[f][cb];
                }
            }
            try
            {
                decoder.SubmitChunkAsync(codeTensor, n, cancel).AsTask().GetAwaiter().GetResult();
            }
            finally
            {
                codeTensor.Dispose();
            }
        }
    }
}
