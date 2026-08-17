using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio.Wake.Speakers;

/// <summary>Answers "who just spoke?" for a captured utterance: CAM++ embedding via <see cref="CamPlusEmbedder"/>,
/// then nearest enrolled centroid by cosine via <see cref="SpeakerProfileStore"/>.
///
/// <para><b>Score the wake word together with the command that follows it.</b> Text-independent speaker verification
/// falls apart on short audio — a system at 0.7% EER on full utterances lands in the several-percent range at around
/// one second — so the wake phrase alone is the weakest possible input. The two-span
/// <see cref="Identify(IBackend, ReadOnlySpan{float}, ReadOnlySpan{float})"/> overload exists to make the joint
/// scoring the easy call; reach for the single-span one only when there genuinely is nothing after the wake word.</para>
///
/// <para>The complementary mitigation is on the enrollment side: enroll <b>text-dependent</b>, on repetitions of the
/// wake phrase itself. Matching content between enrollment and test is what makes a one-second decision usable at
/// all, which is why <see cref="EnrollFromAudio"/> takes the phrase.</para>
///
/// <para>Backends are supplied per call, matching the rest of the engine, so the always-on wake thread can pass its
/// own private <c>CpuBackend</c> rather than contending for a shared one.</para></summary>
public sealed class SpeakerVerifier : IDisposable
{
    /// <summary>Below roughly this much speech the score is dominated by phonetic content rather than by the speaker,
    /// so a match is logged as low-confidence. Not a hard floor — <see cref="CamPlusEmbedder.MinimumSeconds"/> is.</summary>
    public const double ReliableSeconds = 2.0;

    private readonly CamPlusEmbedder _embedder;
    private readonly bool _ownsEmbedder;
    private int _disposed;

    /// <summary>Composes an already-loaded encoder with a store. Set <paramref name="ownsEmbedder"/> false when the
    /// encoder is shared with another subsystem and must outlive this verifier.</summary>
    public SpeakerVerifier(SpeakerProfileStore store, CamPlusEmbedder embedder, bool ownsEmbedder = true)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(embedder);
        Store = store;
        _embedder = embedder;
        _ownsEmbedder = ownsEmbedder;
    }

    /// <summary>Loads CAM++ from disk and pairs it with <paramref name="store"/> (or a default-located one). Throws
    /// an actionable <see cref="InvalidOperationException"/> when the checkpoint is absent — callers that treat
    /// speaker gating as optional should catch it, log, and run without speaker identification rather than refusing
    /// to start the wake listener.</summary>
    public static SpeakerVerifier Load(SpeakerProfileStore? store = null) =>
        new SpeakerVerifier(store ?? new SpeakerProfileStore(), CamPlusEmbedder.Load());

    /// <summary>Whether a CAM++ checkpoint is on disk, so a host can decide whether to offer speaker gating at all
    /// without catching an exception from <see cref="Load"/>.</summary>
    public static bool IsAvailable => CamPlusEmbedder.LocateWeights() is not null;

    /// <summary>The enrolled household. Mutating it (enroll/remove) takes effect on the next identification.</summary>
    public SpeakerProfileStore Store { get; }

    /// <summary>L2-normalized 192-d embedding of one mono 16 kHz clip. Amplitude scale is irrelevant (cepstral mean
    /// normalization removes constant gain), but a single clip must not splice differently-scaled audio together.</summary>
    public float[] Embed(IBackend backend, ReadOnlySpan<float> mono16k) => _embedder.Embed(backend, mono16k);

    /// <summary>Identifies the speaker of the wake phrase <b>plus</b> the command that followed it — the intended
    /// call. Both spans are mono 16 kHz and are scored as one clip, which is what makes the decision survive the
    /// wake phrase's ~1 s of audio. Either span may be empty.</summary>
    public SpeakerMatch Identify(IBackend backend, ReadOnlySpan<float> wakeAudio16k, ReadOnlySpan<float> commandAudio16k)
    {
        if (commandAudio16k.Length == 0)
        {
            return Identify(backend, wakeAudio16k);
        }
        if (wakeAudio16k.Length == 0)
        {
            return Identify(backend, commandAudio16k);
        }
        float[] joined = new float[wakeAudio16k.Length + commandAudio16k.Length];
        wakeAudio16k.CopyTo(joined);
        commandAudio16k.CopyTo(joined.AsSpan(wakeAudio16k.Length));
        return Identify(backend, joined);
    }

    /// <summary>Identifies the speaker of one already-assembled mono 16 kHz utterance. Prefer the two-span overload:
    /// on the wake phrase alone this is the weak case the type remarks warn about. Returns
    /// <see cref="SpeakerMatchOutcome.AudioTooShort"/> rather than throwing when the clip cannot be embedded at all,
    /// because a clipped capture must not take a detection down.
    ///
    /// <para>Scores against <see cref="SpeakerProfileStore.MatchThreshold"/>. For a one-off threshold (a calibration
    /// sweep, a wake word stricter than the household default) call <see cref="Embed"/> and pass the vector to
    /// <see cref="SpeakerProfileStore.Identify(ReadOnlySpan{float}, float)"/> yourself.</para></summary>
    public SpeakerMatch Identify(IBackend backend, ReadOnlySpan<float> utterance16k)
    {
        ArgumentNullException.ThrowIfNull(backend);
        float threshold = Store.MatchThreshold;
        double seconds = utterance16k.Length / (double)CamPlusEmbedder.SampleRate;
        if (utterance16k.Length < CamPlusEmbedder.MinimumSamples)
        {
            Logs.Warning($"[Wake][Speaker] {seconds:0.00}s is below the {CamPlusEmbedder.MinimumSeconds:0.0}s CAM++ floor — treating the speaker as unknown.");
            return new SpeakerMatch(null, 0f, SpeakerMatchOutcome.AudioTooShort, threshold);
        }
        float[] embedding;
        try
        {
            embedding = _embedder.Embed(backend, utterance16k);
        }
        catch (Exception ex)
        {
            Logs.Error($"[Wake][Speaker] Could not embed a {seconds:0.00}s clip; treating the speaker as unknown", ex);
            return new SpeakerMatch(null, 0f, SpeakerMatchOutcome.AudioTooShort, threshold);
        }
        SpeakerMatch match = Store.Identify(embedding, threshold);
        if (seconds < ReliableSeconds)
        {
            Logs.Verbose($"[Wake][Speaker] {match} from only {seconds:0.00}s — scores below {ReliableSeconds:0.0}s are unreliable; "
                + "score the wake word together with the command that follows it.");
        }
        else
        {
            Logs.Verbose($"[Wake][Speaker] {match} from {seconds:0.00}s.");
        }
        return match;
    }

    /// <summary>Enrolls a speaker from raw audio: embeds each utterance, then stores the centroid.
    ///
    /// <para>Pass 3-5 utterances (<see cref="SpeakerProfileStore.RecommendedEnrollmentUtterances"/>), and pass
    /// <paramref name="phrase"/> with the wake phrase when those utterances are repetitions of it — text-dependent
    /// enrollment is the single biggest lever on short-utterance accuracy, and the phrase is recorded so a later
    /// reader knows the profile was built that way.</para></summary>
    public SpeakerProfile EnrollFromAudio(IBackend backend, string name, IReadOnlyList<float[]> utterances16k, string? phrase = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(utterances16k);
        if (utterances16k.Count == 0)
        {
            throw new ArgumentException("Enrollment needs at least one utterance.", nameof(utterances16k));
        }
        List<float[]> embeddings = new List<float[]>(utterances16k.Count);
        for (int i = 0; i < utterances16k.Count; i++)
        {
            float[] utterance = utterances16k[i]
                ?? throw new ArgumentException($"Enrollment utterance {i} is null.", nameof(utterances16k));
            if (utterance.Length < CamPlusEmbedder.MinimumSamples)
            {
                throw new ArgumentException(
                    $"Enrollment utterance {i} is {utterance.Length / (double)CamPlusEmbedder.SampleRate:0.00}s, "
                    + $"below the {CamPlusEmbedder.MinimumSeconds:0.0}s CAM++ floor.", nameof(utterances16k));
            }
            embeddings.Add(_embedder.Embed(backend, utterance));
        }
        return Store.Enroll(name, embeddings, phrase);
    }

    /// <summary>Releases the CAM++ encoder when this verifier owns it.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        if (_ownsEmbedder)
        {
            _embedder.Dispose();
        }
    }
}
