using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Zonos;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Audio.Phonemizer.Espeak;

namespace HartsyInference.Engine.Audio;

/// <summary>Zonos v0.1 (Zyphra/Zonos-v0.1-transformer) — transformer backbone over 9 DAC codebooks → 44.1 kHz voice-cloning TTS, wiring the ResNet293 speaker encoder, the espeak phoneme tokenizer, the prefix conditioner, and the delayed-AR generator. A voice-reference clip is required.</summary>
internal static class ZonosModel
{
    private const string ModelRepo = "Zyphra/Zonos-v0.1-transformer";
    private const string SpeakerRepo = "Zyphra/Zonos-v0.1-speaker-embedding";
    private const string EspeakLanguage = "en-us";

    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => ModelRepo,
        LoadAsync = async (_, _, cancel) =>
        {
            string modelPath = await AudioModelCache.GetAsync(ModelRepo, "model.safetensors", category: "tts", ct: cancel).ConfigureAwait(false);
            // The engine DAC consumes the canonical descript .pth layout (the HF safetensors mirrors are reshaped).
            string dacPath = await AudioModelCache.GetAsync("descript/descript-audio-codec", "weights.pth", category: "tts", ct: cancel).ConfigureAwait(false);
            string speakerPath = await AudioModelCache.GetAsync(SpeakerRepo, "ResNet293_SimAM_ASP_base.pt", category: "tts", ct: cancel).ConfigureAwait(false);
            string ldaPath = await AudioModelCache.GetAsync(SpeakerRepo, "ResNet293_SimAM_ASP_base_LDA-128.pt", category: "tts", ct: cancel).ConfigureAwait(false);

            SafeTensorsLoader modelLoader = new SafeTensorsLoader();
            modelLoader.Load(modelPath);
            PytorchPickleLoader dacLoader = new PytorchPickleLoader();
            dacLoader.Load(dacPath);
            PytorchPickleLoader speakerLoader = new PytorchPickleLoader();
            speakerLoader.Load(speakerPath);
            PytorchPickleLoader ldaLoader = new PytorchPickleLoader();
            ldaLoader.Load(ldaPath);

            EspeakPhonemizer phonemizer = EspeakPhonemizer.FromCache(EspeakLanguage);
            ZonosTts tts = new ZonosTts(ZonosConfig.V0_1Transformer, phonemizer, EspeakLanguage);
            tts.LoadWeights(modelLoader.GetAllTensors(), dacLoader.GetAllTensors(),
                speakerLoader.GetAllTensors(), ldaLoader.GetAllTensors());
            Logs.Info("[Audio][Zonos] Loaded Zyphra/Zonos-v0.1-transformer (ResNet293 clone + DAC, 44.1 kHz).");

            // The speaker encoder wants 16 kHz; the service hands us a mono 24 kHz reference.
            Resampler to16k = Resampler.Create(24_000, tts.SpeakerSampleRate);

            return new TtsRunner(tts.SampleRate, (backend, job) =>
            {
                if (job.ReferenceMono24k is null || job.ReferenceMono24k.Length == 0)
                {
                    throw new NotSupportedException("Zonos clones its speaker — supply a voice-reference clip (there is no random voice).");
                }
                float[] reference16k = to16k.Resample(job.ReferenceMono24k);
                ZonosControls defaults = new ZonosControls();
                ZonosControls controls = new ZonosControls
                {
                    LanguageId = ZonosLanguages.Resolve(EspeakLanguage),
                    CfgScale = job.CfgScale is > 0 ? (float)job.CfgScale.Value : 2.0f,
                    // Reference make_cond_dict ranges: rate 0-40 phonemes/s, pitch_std 0-400.
                    SpeakingRate = job.SpeakingRate is >= 0 and <= 40 ? (float)job.SpeakingRate.Value : defaults.SpeakingRate,
                    PitchStd = job.PitchStd is >= 0 and <= 400 ? (float)job.PitchStd.Value : defaults.PitchStd,
                    Emotion = job.Emotion is { Count: 8 } e ? [.. e.Select(v => (float)v)] : defaults.Emotion,
                };
                return tts.Synthesize(backend, job.Text, reference16k, controls, job.Seed);
            }, tts, modelLoader, dacLoader, speakerLoader, ldaLoader);
        },
    };
}
