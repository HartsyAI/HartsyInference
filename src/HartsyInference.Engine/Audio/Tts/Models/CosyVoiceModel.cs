using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Placement;
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

    /// <summary>The CosyVoice 2 descriptor.</summary>
    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = variant => (variant ?? string.Empty).Contains('/', StringComparison.Ordinal) ? variant! : Repo,
        LoadAsync = async (context, _, cancel) =>
        {
            string llmPath = await AudioModelCache.GetAsync(Repo, "llm.pt", category: "tts", ct: cancel).ConfigureAwait(false);
            string flowPath = await AudioModelCache.GetAsync(Repo, "flow.pt", category: "tts", ct: cancel).ConfigureAwait(false);
            string hiftPath = await AudioModelCache.GetAsync(Repo, "hift.pt", category: "tts", ct: cancel).ConfigureAwait(false);
            string s3genPath = await AudioModelCache.GetAsync(FrozenRepo, "s3gen.safetensors", category: "tts", ct: cancel).ConfigureAwait(false);

            PytorchPickleLoader llmLoader = new PytorchPickleLoader();
            llmLoader.Load(llmPath);
            PytorchPickleLoader flowLoader = new PytorchPickleLoader();
            flowLoader.Load(flowPath);
            PytorchPickleLoader hiftLoader = new PytorchPickleLoader();
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

            Qwen2Tokenizer tokenizer = new Qwen2Tokenizer();
            Logs.Info("[Audio][CosyVoice] Loaded FunAudioLLM/CosyVoice2-0.5B (Qwen LM + OT-CFM flow + HiFTNet, 24 kHz)"
                + $"{(lm.Placement is not null ? ", layer-split across " + lm.Placement.Stages.Count + " GPUs" : "")}.");

            IDisposable?[] keep = [pipeline, llmLoader, flowLoader, hiftLoader, s3genLoader];
            float[] Synth(IBackend backend, TtsJob job)
            {
                if (job.ReferenceMono24k is null || job.ReferenceMono24k.Length == 0)
                {
                    throw new InvalidOperationException(
                        "CosyVoice 2 is zero-shot — it needs a voice reference. Supply a short WAV clip as the request's reference.");
                }
                int[] textTokenIds = [.. tokenizer.EncodeRawByteLevel(job.Text)];
                int[] referenceTextTokens = string.IsNullOrWhiteSpace(job.RefText) ? [] : [.. tokenizer.EncodeRawByteLevel(job.RefText)];
                // The pipeline derives the S3 mel, CAM++ fbank, and flow mel from the raw reference itself.
                return pipeline.Synthesize(backend, textTokenIds, referenceAudio: job.ReferenceMono24k,
                    referenceSampleRate: 24_000, referenceTextTokens: referenceTextTokens, seed: job.Seed);
            }
            IAsyncEnumerable<AudioChunk> Stream(IBackend backend, TtsJob job, CancellationToken cancel)
            {
                if (job.ReferenceMono24k is null || job.ReferenceMono24k.Length == 0)
                {
                    throw new InvalidOperationException(
                        "CosyVoice 2 is zero-shot — it needs a voice reference. Supply a short WAV clip as the request's reference.");
                }
                int[] textTokenIds = [.. tokenizer.EncodeRawByteLevel(job.Text)];
                int[] referenceTextTokens = string.IsNullOrWhiteSpace(job.RefText) ? [] : [.. tokenizer.EncodeRawByteLevel(job.RefText)];
                return pipeline.SynthesizeStream(backend, textTokenIds, referenceAudio: job.ReferenceMono24k,
                    referenceSampleRate: 24_000, referenceTextTokens: referenceTextTokens, seed: job.Seed, cancel: cancel);
            }
            return new StreamingTtsRunner(config.SampleRate, Synth, Stream, keep);
        },
    };

    /// <summary>Plans the Qwen2.5-0.5B backbone's layer split across the context's shard devices (explicit ratios
    /// win, else proportional to live free VRAM) and binds each stage to its resolved backend. Mirrors
    /// <c>YueMusicModel.BuildStage1Placement</c>.</summary>
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
