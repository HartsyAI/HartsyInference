using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>F5-TTS v1 Base (SWivid/F5-TTS) — zero-shot voice-clone TTS: a flow-matching DiT denoises a target mel in
/// the voice of a reference clip, vocoded by Vocos to 24 kHz. The DiT, char vocab, and vocoder auto-download.
///
/// <para>Zero-shot: the caller must supply BOTH the reference clip and its transcript. The pipeline's
/// <c>GenerateFromAudio</c> owns the exact F5/Vocos mel front-end, so the mel is never built here.</para></summary>
internal static class F5TtsModel
{
    private const string Repo = "SWivid/F5-TTS";

    /// <summary>The F5-TTS descriptor.</summary>
    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = variant => (variant ?? string.Empty).Contains('/', StringComparison.Ordinal) ? variant! : Repo,
        LoadAsync = async (_, _, cancel) =>
        {
            F5TtsPipeline pipeline = await F5TtsPipeline.LoadAsync(ct: cancel).ConfigureAwait(false);
            Logs.Info("[Audio][F5-TTS] Loaded SWivid/F5-TTS v1 Base (DiT + Vocos 24 kHz).");

            return new TtsRunner(24_000, (backend, job) =>
            {
                if (job.ReferenceMono24k is null || job.ReferenceMono24k.Length == 0)
                {
                    throw new InvalidOperationException(
                        "F5-TTS is zero-shot — it needs a voice reference clip plus the words spoken in it (reference text).");
                }
                if (string.IsNullOrWhiteSpace(job.RefText))
                {
                    throw new InvalidOperationException(
                        "F5-TTS needs the transcript of the reference clip — it aligns the cloned voice against that text.");
                }
                return pipeline.GenerateFromAudio(backend, job.ReferenceMono24k, 24_000, job.RefText, job.Text,
                    new F5TtsOptions
                    {
                        Seed = (ulong)job.Seed,
                        Steps = job.NfeStep ?? 32,
                        CfgStrength = job.CfgScale.HasValue ? (float)job.CfgScale.Value : 2.0f,
                        Speed = job.Speed.HasValue ? (float)job.Speed.Value : 1.0f,
                    });
            }, pipeline);
        },
    };
}
