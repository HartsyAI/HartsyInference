using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;
using HartsyInference.Audio.Phonemizer.Espeak;

namespace HartsyInference.Engine.Audio;

/// <summary>StyleTTS 2 (yl4579/StyleTTS2-LibriTTS) — diffusion-style TTS at 24 kHz. Voice-clone: the reference clip's
/// 256-d style comes from the StarGAN-v2 style encoders, then the shared PLBERT/text-encoder/prosody backbone plus
/// the LibriTTS HiFi-GAN generator synthesize the target text in that voice. Random (no-reference) synthesis needs
/// the diffusion style sampler, which is not yet reconciled to the real checkpoint, so a reference is required.</summary>
internal static class StyleTts2Model
{
    private const string Repo = "yl4579/StyleTTS2-LibriTTS";
    private const string CheckpointFile = "Models/LibriTTS/epochs_2nd_00020.pth";
    // Trained on American-English (espeak en-us) phonemes WITH punctuation preserved: British "en" gives wrong
    // vowels and stripping punctuation makes the prosody predictor slur across phrase boundaries.
    private const string EspeakLanguage = "en-us";

    /// <summary>The StyleTTS 2 descriptor.</summary>
    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => Repo,
        LoadAsync = async (_, cancel) =>
        {
            string checkpoint = await AudioModelCache.GetAsync(Repo, CheckpointFile, ct: cancel).ConfigureAwait(false);
            StyleTts2Pipeline pipeline = StyleTts2Pipeline.LoadFromCheckpoint(checkpoint);
            EspeakPhonemizer phonemizer = EspeakPhonemizer.FromCache(EspeakLanguage);
            Logs.Info("[Audio][StyleTTS2] Loaded yl4579/StyleTTS2-LibriTTS (StarGAN-v2 clone + HiFiGAN, 24 kHz).");

            string Ipa(string text) =>
                string.Join(' ', phonemizer.PhonemizeToIpa(text, EspeakLanguage, preservePunctuation: true)
                    .Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));

            return new TtsRunner(24_000, (backend, job) =>
            {
                if (job.ReferenceMono24k is null || job.ReferenceMono24k.Length == 0)
                {
                    throw new NotSupportedException(
                        "StyleTTS2 clones its speaker — supply a voice-reference clip. Random/unconditional synthesis "
                        + "needs the diffusion style sampler, which is not yet reconciled to the real checkpoint.");
                }
                float speed = (float)(job.Speed ?? 1.0);
                return pipeline.SynthesizeCloneFromAudio(backend, Ipa(job.Text), job.ReferenceMono24k, 24_000, speed);
            }, pipeline);
        },
    };
}
