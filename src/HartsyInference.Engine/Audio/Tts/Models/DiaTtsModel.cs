using System.Runtime.CompilerServices;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Models.Dia;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.PyTorch;

namespace HartsyInference.Engine.Audio;

/// <summary>Dia-1.6B (0626 release) — byte-level two-speaker dialogue TTS at 44.1 kHz with the Descript DAC codec.
///
/// <para>DAC's decoder uses symmetric (non-causal) padding throughout — verified against the real weights during
/// research, no causal variant exists — so unlike Mimi/Firefly it genuinely needs both past AND future code-frame
/// context, and a windowed decode can't just carry left-context state the way Mimi's can. Streaming here follows
/// nari-labs' own streaming attempt for the ORIGINAL Dia (DAC-based; their successor <c>Dia2</c> replaced DAC with
/// Mimi specifically to get real streaming instead): full re-decode of the whole utterance-so-far on every
/// emission, keeping only the new trailing samples — see <see cref="DiaPipeline.DecodeSettledFramesTail"/>.</para></summary>
internal static class DiaTtsModel
{
    private const string ModelRepo = "nari-labs/Dia-1.6B-0626";
    private const string DacRepo = "descript/descript-audio-codec";
    private const int FramesPerChunk = 25; // Dia frame rate for the DAC-44.1kHz codec; ~similar order to the others.

    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        // The 0626 release, NOT the original Dia-1.6B: same architecture/keys/shapes, but the original checkpoint's
        // weights degenerate through the engine while 0626 produces the full multi-turn dialogue word-correct.
        ResolveRepo = _ => ModelRepo,
        LoadAsync = async (_, _, cancel) =>
        {
            string modelPath = await AudioModelCache.GetAsync(ModelRepo, "pytorch_model.bin", category: "tts", ct: cancel).ConfigureAwait(false);
            // The canonical descript .pth has the layout the engine expects (the HF safetensors mirrors are reshaped).
            string dacPath = await AudioModelCache.GetAsync(DacRepo, "weights.pth", category: "tts", ct: cancel).ConfigureAwait(false);
            // 0626 is a flat pickle state_dict; non-recursive flatten keeps the encoder./decoder. prefixes intact.
            PytorchPickleLoader modelLoader = new PytorchPickleLoader();
            modelLoader.Load(modelPath, recursiveFlatten: false);
            PytorchPickleLoader dacLoader = new PytorchPickleLoader();
            dacLoader.Load(dacPath);

            DiaPipeline pipeline = new DiaPipeline(DiaConfig.Dia1_6B);
            pipeline.LoadWeights(modelLoader.GetAllTensors(), dacLoader.GetAllTensors());
            Logs.Info("[Audio][Dia] Loaded nari-labs/Dia-1.6B-0626 (byte-level dialogue TTS, 44.1 kHz).");

            Session session = new(pipeline);
            return new StreamingTtsRunner(44_100, session.Synthesize, session.SynthesizeStream,
                pipeline, modelLoader, dacLoader);
        },
    };

    private sealed class Session(DiaPipeline pipeline)
    {
        private static string BuildText(string text)
        {
            // Dia was trained on [S1]/[S2]-tagged dialogue; untagged text degenerates into repetition loops.
            string tagged = text.Contains("[S", StringComparison.Ordinal) ? text : $"[S1] {text}";
            if (tagged.Length < 120)
            {
                Logs.Warning($"[Audio][Dia] Prompt is very short ({tagged.Length} chars) — Dia-1.6B tends to produce "
                    + "silence below ~2 sentences (upstream behaves the same). Use longer dialogue-style text.");
            }
            return tagged;
        }

        public float[] Synthesize(IBackend backend, TtsJob job)
            => pipeline.Generate(backend, AudioTextFrontend.DiaBytes(BuildText(job.Text)),
                job.MaxTokens is > 0 ? job.MaxTokens.Value : 1720, job.Seed, null,
                job.CfgScale, job.TopK, job.Temperature, job.TopP);

        public async IAsyncEnumerable<AudioChunk> SynthesizeStream(IBackend backend, TtsJob job, [EnumeratorCancellation] CancellationToken cancel)
        {
            int[] textBytes = AudioTextFrontend.DiaBytes(BuildText(job.Text));
            int maxTokens = job.MaxTokens is > 0 ? job.MaxTokens.Value : 1720;
            using AudioStreamer streamer = new();
            long samplesEmitted = 0;

            Task producer = Task.Run(() =>
            {
                List<int[]> allSettled = new(maxTokens);
                int lastDecodedFrame = 0;
                try
                {
                    pipeline.Generate(backend, textBytes, maxTokens, job.Seed, null,
                        job.CfgScale, job.TopK, job.Temperature, job.TopP, onSettledFrame: frame =>
                        {
                            allSettled.Add(frame);
                            if (allSettled.Count - lastDecodedFrame < FramesPerChunk) return;
                            EmitTail(backend, streamer, allSettled, cancel, ref samplesEmitted);
                            lastDecodedFrame = allSettled.Count;
                        });
                    if (allSettled.Count > lastDecodedFrame)
                    {
                        EmitTail(backend, streamer, allSettled, cancel, ref samplesEmitted);
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

        private void EmitTail(IBackend backend, AudioStreamer streamer, List<int[]> allSettled, CancellationToken cancel, ref long samplesEmitted)
        {
            float[] tail = pipeline.DecodeSettledFramesTail(backend, allSettled, (int)samplesEmitted);
            if (tail.Length == 0) return;
            AudioChunk chunk = new(tail, 44_100, Channels: 1, samplesEmitted);
            samplesEmitted += tail.Length;
            streamer.Put(chunk, cancel).AsTask().GetAwaiter().GetResult();
        }
    }
}
