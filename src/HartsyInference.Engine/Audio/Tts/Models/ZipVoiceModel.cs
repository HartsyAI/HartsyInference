using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>ZipVoice (k2-fsa/ZipVoice) — zero-shot voice-clone TTS: flow-matching Euler sampling under a
/// Zipformer (fm_decoder + text_encoder), vocoded via the shared Vocos port. English-only, single checkpoint —
/// there is no variant to resolve. The DiT, char vocab, and vocoder auto-download.
///
/// <para>Zero-shot: the caller must supply both the reference clip and its transcript, same contract as F5-TTS.</para></summary>
internal static class ZipVoiceModel
{
    private const string Repo = "k2-fsa/ZipVoice";

    /// <summary>The ZipVoice descriptor.</summary>
    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => Repo,
        LoadAsync = async (_, _, cancel) =>
        {
            ZipVoicePipeline pipeline = await ZipVoicePipeline.LoadAsync(ct: cancel).ConfigureAwait(false);
            Logs.Info("[Audio][ZipVoice] Loaded k2-fsa/ZipVoice (Zipformer flow-matching + Vocos 24 kHz).");

            return new TtsRunner(24_000, (backend, job) =>
            {
                if (job.ReferenceMono24k is null || job.ReferenceMono24k.Length == 0)
                {
                    throw new InvalidOperationException(
                        "ZipVoice is zero-shot — it needs a voice reference clip plus the words spoken in it (reference text).");
                }
                if (string.IsNullOrWhiteSpace(job.RefText))
                {
                    throw new InvalidOperationException(
                        "ZipVoice needs the transcript of the reference clip — it aligns the cloned voice against that text.");
                }
                return pipeline.GenerateFromAudio(backend, job.ReferenceMono24k, 24_000, job.RefText, job.Text,
                    new ZipVoiceOptions
                    {
                        Seed = (ulong)job.Seed,
                        Speed = job.Speed.HasValue ? (float)job.Speed.Value : 1.0f,
                    });
            }, pipeline);
        },
    };
}
