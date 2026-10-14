using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Models.GptSoVits;
using HartsyInference.Audio.Models.Hubert;
using HartsyInference.Audio.Models.OpenVoice;
using HartsyInference.Audio.Models.Vits;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;

namespace HartsyInference.Engine.Audio;

/// <summary>GPT-SoVITS v2 (lj1995/GPT-SoVITS) — zero-shot ENGLISH TTS: HuBERT SSL + Text2Semantic (s1, AR) + SoVITS
/// (s2) at 32 kHz, in the reference speaker's voice. Needs a reference clip (decoded to 16 kHz for HuBERT and 32 kHz
/// for the 1025-bin linear spectrogram) and its transcript; the English path uses zero BERT.
///
/// <para>Caveat: the s1 AR loop has no KV cache (O(n²)) in the engine today, so long sentences are slow.</para></summary>
internal static class GptSoVitsModel
{
    private const string Repo = "lj1995/GPT-SoVITS";
    private const string S2File = "gsv-v2final-pretrained/s2G2333k.pth";
    private const string S1File = "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt";
    private const string HubertFile = "chinese-hubert-base/pytorch_model.bin";

    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => Repo,
        LoadAsync = async (_, _, cancel) =>
        {
            string s2Path = await AudioModelCache.GetAsync(Repo, S2File, category: "tts", ct: cancel).ConfigureAwait(false);
            string s1Path = await AudioModelCache.GetAsync(Repo, S1File, category: "tts", ct: cancel).ConfigureAwait(false);
            string hubertPath = await AudioModelCache.GetAsync(Repo, HubertFile, category: "tts", ct: cancel).ConfigureAwait(false);

            AnyFormatCheckpointLoader s2Loader = new AnyFormatCheckpointLoader();
            s2Loader.Load(s2Path);
            IReadOnlyDictionary<string, Tensor> s2Weights = s2Loader.GetAllTensors();
            SoVitsSynthesizer s2 = new SoVitsSynthesizer(V2Config(), sslDim: 768, sslLayers: 3, textLayers: 6,
                enc2Layers: 3, mrteHidden: 512, mrteHeads: 4);
            s2.LoadWeights(s2Weights);
            SoVitsRefEnc referenceEncoder = new SoVitsRefEnc();
            referenceEncoder.LoadWeights(s2Weights, "ref_enc");

            AnyFormatCheckpointLoader s1Loader = new AnyFormatCheckpointLoader();
            s1Loader.Load(s1Path);
            Text2Semantic s1 = new Text2Semantic(new Text2SemanticConfig());
            s1.LoadWeights(s1Loader.GetAllTensors(), "model");

            AnyFormatCheckpointLoader hubertLoader = new AnyFormatCheckpointLoader();
            hubertLoader.Load(hubertPath);
            Hubert hubert = new Hubert(new HubertConfig());
            hubert.LoadWeights(hubertLoader.GetAllTensors());

            GptSoVitsPipeline pipeline = new GptSoVitsPipeline(hubert, s1, s2, referenceEncoder);
            Logs.Info("[Audio][GPT-SoVITS] Loaded lj1995/GPT-SoVITS v2 (HuBERT + s1 + s2, 32 kHz, English zero-shot).");

            // Components are shared resources the pipeline does not own → dispose them (and the loaders) here.
            IDisposable?[] keep = [pipeline, hubert, s1, s2, s2Loader, s1Loader, hubertLoader];
            return new TtsRunner(pipeline.SampleRate, (backend, job) =>
            {
                if (job.Reference is null || job.Reference.Data.Length == 0)
                {
                    throw new InvalidOperationException(
                        "GPT-SoVITS needs a reference voice clip — supply a short WAV (~3-10s) as the request's reference.");
                }
                if (string.IsNullOrWhiteSpace(job.RefText))
                {
                    throw new InvalidOperationException(
                        "GPT-SoVITS needs the reference transcript — the exact words spoken in the reference clip.");
                }

                // Reference clip → 16 kHz (HuBERT) + 32 kHz (1025-bin linear spectrogram for the speaker embedding).
                float[] reference16k = AudioClipCodec.DecodeMono(job.Reference, 16_000);
                float[] reference32k = AudioClipCodec.DecodeMono(job.Reference, 32_000);
                // Upstream TTS._set_prompt_semantic appends int(sampling_rate × 0.3) zeros, sized at the model's 32 kHz
                // rate, to the 16 kHz clip before HuBERT: 0.6 s of silence at 16 kHz.
                int hubertLength = reference16k.Length + pipeline.SampleRate * 3 / 10;
                Tensor referencePcm = new Tensor(new TensorShape(1, 1, hubertLength), DType.F32);
                Span<float> hubertInput = referencePcm.AsSpan<float>();
                reference16k.AsSpan().CopyTo(hubertInput);
                hubertInput[reference16k.Length..].Clear();
                Tensor referenceSpec = LinearSpectrogram.Extract(reference32k, 2048, 640);
                int specFrames = (int)referenceSpec.Shape[2];

                // Upstream TTS.run: the target is cut (four sentences per segment), each segment is synthesized after
                // the reference transcript's phones (s1 text span = ref + segment, zero BERT on the English path), and
                // the fragments are joined with 0.3 s of silence after each.
                int[] referenceIds = GptSoVitsSymbols.ToSequence(GptSoVitsTextSegmenter.PrepareEnglishPrompt(job.RefText).Phones);
                List<GptSoVitsTextSegment> segments = GptSoVitsTextSegmenter.PrepareEnglish(job.Text);
                if (segments.Count == 0)
                {
                    throw new InvalidOperationException("GPT-SoVITS found nothing to speak in the text.");
                }
                int gap = pipeline.SampleRate * 3 / 10;
                List<float> output = [];
                try
                {
                    for (int i = 0; i < segments.Count; i++)
                    {
                        int[] targetIds = GptSoVitsSymbols.ToSequence(segments[i].Phones);
                        int[] allIds = [.. referenceIds, .. targetIds];
                        using Tensor zeroBert = new Tensor(new TensorShape(1024, allIds.Length), DType.F32);
                        float[] fragment = pipeline.Generate(backend, referencePcm, hubertLength, referenceSpec,
                            specFrames, allIds, zeroBert, targetIds, seed: job.Seed == 0 ? 0 : job.Seed + i);
                        AppendPeakLimited(output, fragment);
                        output.AddRange(new float[gap]);
                    }
                    return [.. output];
                }
                finally
                {
                    referencePcm.Dispose();
                    referenceSpec.Dispose();
                }
            }, keep);
        },
    };

    /// <summary>Appends <paramref name="fragment"/>, scaled down to peak 1 when it clips (upstream audio_postprocess).</summary>
    private static void AppendPeakLimited(List<float> output, float[] fragment)
    {
        float peak = 0f;
        foreach (float s in fragment) peak = MathF.Max(peak, MathF.Abs(s));
        float divisor = peak > 1f ? peak : 1f;
        foreach (float s in fragment) output.Add(s / divisor);
    }

    /// <summary>GPT-SoVITS v2 SoVITS (VITS) config — resblock "1", upsample product 640 hop, 32 kHz output.</summary>
    private static VitsConfig V2Config() => new VitsConfig
    {
        InterChannels = 192,
        HiddenChannels = 192,
        FilterChannels = 768,
        NumHeads = 2,
        NumEncoderLayers = 6,
        EncoderKernelSize = 3,
        WindowSize = 4,
        GinChannels = 512,
        FlowLayers = 4,
        FlowFlows = 4,
        FlowKernelSize = 5,
        FlowDilationRate = 1,
        ResBlock = "1",
        ResBlockKernelSizes = [3, 7, 11],
        ResBlockDilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        UpsampleRates = [10, 8, 2, 2, 2],
        UpsampleInitialChannel = 512,
        UpsampleKernelSizes = [16, 16, 8, 2, 2],
        SampleRate = 32_000,
    };
}
