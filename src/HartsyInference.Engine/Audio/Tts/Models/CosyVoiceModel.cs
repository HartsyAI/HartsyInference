using System.Runtime.CompilerServices;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Frontends.WeText;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Placement;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>CosyVoice 2 (FunAudioLLM/CosyVoice2-0.5B) — zero-shot TTS: a Qwen2.5-0.5B LM emits S3 speech tokens that
/// an OT-CFM flow turns into a mel, vocoded by HiFTNet to 24 kHz. Speaker identity comes from a reference clip.
///
/// <para>The frozen CAM++ and S3 encoders ship only as ONNX upstream (fused Conv+BN, mangled names), so they are
/// loaded from ResembleAI/chatterbox's clean-named safetensors of the SAME frozen models.</para></summary>
internal static class CosyVoiceModel
{
    private const string Repo = "FunAudioLLM/CosyVoice2-0.5B";
    private const string FrozenRepo = "ResembleAI/chatterbox";
    private const string BlankEnVocab = "CosyVoice-BlankEN/vocab.json";
    private const string BlankEnMerges = "CosyVoice-BlankEN/merges.txt";

    /// <summary>Shared by the loader and prefetch. The LM is last — it is the artifact that marks CosyVoice
    /// installed — and the S3 generator comes from the frozen Chatterbox repo rather than this model's own.</summary>
    private static readonly AudioModelFile[] ModelFiles =
    [
        new("s3gen.safetensors", Repo: FrozenRepo),
        // CosyVoice 2's own Qwen vocabulary: its CJK merges differ from the embedded Qwen3 set.
        new(BlankEnVocab),
        new(BlankEnMerges),
        new("flow.pt"),
        new("hift.pt"),
        new("llm.pt"),
    ];

    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = variant => (variant ?? string.Empty).Contains('/', StringComparison.Ordinal) ? variant! : Repo,
        ResolveFiles = (_, _) => Task.FromResult<IReadOnlyList<AudioModelFile>>(ModelFiles),
        ValidateReference = reference => CosyVoiceReferenceGuard.Check(reference.Length, SpeechService.ReferenceSampleRate),
        LoadAsync = async (context, _, cancel) =>
        {
            IReadOnlyDictionary<string, string> fetched = await AudioModelCache
                .FetchAllAsync(Repo, ModelFiles, category: "tts", ct: cancel).ConfigureAwait(false);
            string llmPath = fetched["llm.pt"];
            string flowPath = fetched["flow.pt"];
            string hiftPath = fetched["hift.pt"];
            string s3genPath = fetched["s3gen.safetensors"];

            AnyFormatCheckpointLoader llmLoader = new AnyFormatCheckpointLoader();
            llmLoader.Load(llmPath);
            AnyFormatCheckpointLoader flowLoader = new AnyFormatCheckpointLoader();
            flowLoader.Load(flowPath);
            AnyFormatCheckpointLoader hiftLoader = new AnyFormatCheckpointLoader();
            hiftLoader.Load(hiftPath);
            SafeTensorsLoader s3genLoader = new SafeTensorsLoader();
            s3genLoader.Load(s3genPath);
            Dictionary<string, Tensor> s3gen = s3genLoader.GetAllTensors();

            CosyVoiceConfig config = CosyVoiceConfig.V2_0_5B;
            CosyVoiceQwenLm lm = new CosyVoiceQwenLm(config);
            lm.LoadWeights(llmLoader.GetAllTensors());
            if (context.IsSharded)
            {
                lm.Placement = BuildLlmPlacement(context, lm, config);
            }
            CosyVoiceFlow flow = new CosyVoiceFlow(config);
            flow.LoadWeights(flowLoader.GetAllTensors());
            HiFTNetVocoder vocoder = new HiFTNetVocoder(config.Hift);
            vocoder.LoadWeights(hiftLoader.GetAllTensors());
            CamPlusSpeakerEncoder speaker = new CamPlusSpeakerEncoder(config.Flow.SpeakerEmbedDim);
            speaker.LoadWeights(s3gen, "speaker_encoder");
            S3Tokenizer s3 = new S3Tokenizer();
            s3.LoadWeights(s3gen, "tokenizer");
            CosyVoicePipeline pipeline = new CosyVoicePipeline(config, lm, flow, vocoder, speaker, s3);

            Qwen2Tokenizer tokenizer = new Qwen2Tokenizer(fetched[BlankEnVocab], fetched[BlankEnMerges]);
            int[] Encode(string text) => tokenizer.EncodeExact(text, CosyVoiceTextFrontend.SpecialTokens);
            // The reference pins wetext 0.0.4 and builds both its English and Chinese normalizers with the default
            // (auto-detecting) language, so one Auto normalizer serves both. Where wetext raises, the reference
            // request fails; this keeps the raw text and logs instead.
            WeTextNormalizer weText = await WeTextAssets.LoadAsync(WeTextLanguage.Auto, removeErhua: false, cancel).ConfigureAwait(false);
            string Tn(string text)
            {
                if (weText.TryNormalize(text, out string? normalized)) return normalized!;
                Logs.Warning($"[Audio][CosyVoice] Text normalization found no path for '{text}'; using it unnormalized.");
                return text;
            }
            CosyVoiceTextFrontend frontend = new CosyVoiceTextFrontend(text => Encode(text).Length, Tn, Tn);
            Logs.Info("[Audio][CosyVoice] Loaded FunAudioLLM/CosyVoice2-0.5B (Qwen LM + OT-CFM flow + HiFTNet, 24 kHz)"
                + $"{(lm.Placement is not null ? ", layer-split across " + lm.Placement.Stages.Count + " GPUs" : "")}.");

            IDisposable?[] keep = [pipeline, llmLoader, flowLoader, hiftLoader, s3genLoader];

            // Reference flow (cosyvoice/cli/cosyvoice.py inference_zero_shot): the prompt transcript is normalized
            // without splitting, the target text is normalized and split into segments, and each segment runs its own
            // LM/flow/vocoder pass with the same voice prompt.
            (List<int[]> Segments, int[] Reference) Prepare(TtsJob job)
            {
                if (job.ReferenceMono24k is null || job.ReferenceMono24k.Length == 0)
                {
                    throw new InvalidOperationException(
                        "CosyVoice 2 is zero-shot — it needs a voice reference. Supply a short WAV clip as the request's reference.");
                }
                List<int[]> segments = [.. frontend.Segments(job.Text).Select(Encode).Where(static ids => ids.Length > 0)];
                if (segments.Count == 0)
                {
                    throw new InvalidOperationException("CosyVoice 2 found nothing to speak in the text.");
                }
                int[] reference = string.IsNullOrWhiteSpace(job.RefText) ? [] : Encode(frontend.Normalize(job.RefText));
                return (segments, reference);
            }
            float[] Synth(IBackend backend, TtsJob job)
            {
                (List<int[]> segments, int[] reference) = Prepare(job);
                List<float[]> parts = new(segments.Count);
                foreach (int[] ids in segments)
                {
                    // The pipeline derives the S3 mel, CAM++ fbank, and flow mel from the raw reference itself.
                    parts.Add(pipeline.Synthesize(backend, ids, referenceAudio: job.ReferenceMono24k,
                        referenceSampleRate: 24_000, referenceTextTokens: reference, seed: job.Seed));
                }
                float[] audio = new float[parts.Sum(static p => p.Length)];
                int offset = 0;
                foreach (float[] part in parts)
                {
                    part.CopyTo(audio, offset);
                    offset += part.Length;
                }
                return audio;
            }
            async IAsyncEnumerable<AudioChunk> Stream(IBackend backend, TtsJob job, [EnumeratorCancellation] CancellationToken cancel)
            {
                (List<int[]> segments, int[] reference) = Prepare(job);
                long offset = 0;
                foreach (int[] ids in segments)
                {
                    long segmentStart = offset;
                    await foreach (AudioChunk chunk in pipeline.SynthesizeStream(backend, ids, referenceAudio: job.ReferenceMono24k,
                        referenceSampleRate: 24_000, referenceTextTokens: reference, seed: job.Seed, cancel: cancel).ConfigureAwait(false))
                    {
                        // Each segment's stream counts from zero; re-base it onto the request's timeline.
                        yield return chunk with { StartSampleOffset = segmentStart + chunk.StartSampleOffset };
                        offset = Math.Max(offset, segmentStart + chunk.StartSampleOffset + chunk.FrameCount);
                    }
                }
            }
            return new StreamingTtsRunner(config.SampleRate, Synth, Stream, keep);
        },
    };

    /// <summary>Plans the Qwen2.5-0.5B backbone's layer split across the context's shard devices (explicit ratios win, else proportional to live free VRAM) and binds each stage to its resolved backend. Mirrors <c>YueMusicModel.BuildStage1Placement</c>.</summary>
    private static LlmPlacement BuildLlmPlacement(TtsLoadContext context, CosyVoiceQwenLm lm, CosyVoiceConfig config)
    {
        int layers = config.Llm.NumHiddenLayers;
        long totalBytes = 0;
        foreach (Tensor tensor in lm.EnumerateWeights())
        {
            totalBytes += Tensor.ComputeByteSize(tensor.Shape, tensor.DType);
        }
        string[] devices = [.. context.ShardStages!.Select(s => s.Selector)];
        IReadOnlyList<LlmStagePlan> plan = PlacementPlanner.LlmSplitPlan(
            devices, context.ShardRatios, layers, totalBytes / Math.Max(1, layers));
        List<LlmStage> stages = new(plan.Count);
        foreach (LlmStagePlan stagePlan in plan)
        {
            IBackend backend = context.ShardStages!.First(s => s.Selector == stagePlan.Device).Backend;
            stages.Add(new LlmStage(backend, stagePlan.StartLayer, stagePlan.EndLayer));
        }
        Logs.Info($"[Audio][CosyVoice] LM layer split: "
            + string.Join(" + ", plan.Select(p => $"{p.Device}[{p.StartLayer},{p.EndLayer})")) + ".");
        return new LlmPlacement([.. stages]);
    }
}
