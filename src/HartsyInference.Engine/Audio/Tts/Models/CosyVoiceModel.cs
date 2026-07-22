using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
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
        LoadAsync = async (_, cancel) =>
        {
            string llmPath = await AudioModelCache.GetAsync(Repo, "llm.pt", ct: cancel).ConfigureAwait(false);
            string flowPath = await AudioModelCache.GetAsync(Repo, "flow.pt", ct: cancel).ConfigureAwait(false);
            string hiftPath = await AudioModelCache.GetAsync(Repo, "hift.pt", ct: cancel).ConfigureAwait(false);
            string s3genPath = await AudioModelCache.GetAsync(FrozenRepo, "s3gen.safetensors", ct: cancel).ConfigureAwait(false);

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
            Logs.Info("[Audio][CosyVoice] Loaded FunAudioLLM/CosyVoice2-0.5B (Qwen LM + OT-CFM flow + HiFTNet, 24 kHz).");

            IDisposable?[] keep = [pipeline, llmLoader, flowLoader, hiftLoader, s3genLoader];
            return new TtsRunner(config.SampleRate, (backend, job) =>
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
            }, keep);
        },
    };
}
