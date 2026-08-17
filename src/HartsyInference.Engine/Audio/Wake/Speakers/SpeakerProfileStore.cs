using System.Text;
using System.Text.Json;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio.Wake.Speakers;

/// <summary>The enrolled household, persisted under <c>{wake model root}/speakers/</c> as one JSON sidecar plus one
/// binary embedding file per speaker, and reloaded on construction so enrollment survives a restart.
///
/// <para>Deliberately knows nothing about audio or CAM++: it stores embeddings and answers
/// <see cref="Identify(ReadOnlySpan{float})"/> from vectors alone. <see cref="SpeakerVerifier"/> is the half that
/// turns microphone audio into those vectors. Keeping the split means the enrollment maths and the open-set decision
/// are exercised without weights, and a host that computes embeddings some other way can still use this store.</para>
///
/// <para>The centroid is always re-derived from the stored enrollment embeddings rather than persisted alongside
/// them, so there is no second copy that can silently drift out of agreement with the utterances it claims to
/// summarize.</para>
///
/// <para>Thread-safe: the always-on wake thread reads it while an enrollment API call may be writing.</para></summary>
public sealed class SpeakerProfileStore
{
    /// <summary>Household convention from the Odyssey 2022 baselines (arXiv:2205.00288): 3-5 utterances per speaker.</summary>
    public const int RecommendedEnrollmentUtterances = 3;

    /// <summary>Cosine similarity at or above which the nearest centroid is accepted as that speaker.
    ///
    /// <para><b>Uncalibrated.</b> No equal-error rate has been measured for CAM++ centroid scoring on this codebase,
    /// let alone on the microphones and rooms of a given house — this is the middle of the 0.25-0.5 band the
    /// household-speaker-recognition literature reports for cosine-to-centroid open-set scoring, chosen so that the
    /// short-utterance case (a wake phrase plus a one-second command, where scores are depressed relative to full
    /// utterances) is not rejected outright. Treat it as a starting point for a real trial, not a validated number.
    /// Raise it to cut false accepts, lower it to cut false rejects; the log line on every match carries the score
    /// and the nearest name precisely so those trials can be run from production logs.</para></summary>
    public const float DefaultMatchThreshold = 0.35f;

    private const string EmbeddingMagic = "HSPK";
    private const int EmbeddingFormatVersion = 1;
    private const string JsonExtension = ".json";
    private const string EmbeddingExtension = ".emb";

    private readonly object _gate = new object();
    private readonly Dictionary<string, SpeakerProfile> _profiles = new Dictionary<string, SpeakerProfile>(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string> _stems = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
    private readonly string _root;
    private float _threshold;

    /// <summary>Loads every profile already on disk. <paramref name="directory"/> defaults to
    /// <c>{models}/audio/wake/speakers</c>; pass the wake service's own model root plus <c>speakers</c> when it has
    /// been overridden. A profile that fails to load is logged and skipped, never fatal — one corrupt sidecar must
    /// not take the whole household down.</summary>
    public SpeakerProfileStore(string? directory = null, float matchThreshold = DefaultMatchThreshold)
    {
        _root = string.IsNullOrWhiteSpace(directory) ? DefaultDirectory() : Path.GetFullPath(directory);
        MatchThreshold = matchThreshold;
        Reload();
    }

    /// <summary>Where the sidecars live; created lazily on the first enrollment.</summary>
    public string RootDirectory => _root;

    /// <summary>Open-set acceptance threshold, see <see cref="DefaultMatchThreshold"/> for why it is a guess.</summary>
    public float MatchThreshold
    {
        get => _threshold;
        set
        {
            if (value is < -1f or > 1f || float.IsNaN(value))
            {
                throw new ArgumentOutOfRangeException(nameof(value), value, "A cosine threshold must lie in [-1, 1].");
            }
            _threshold = value;
        }
    }

    /// <summary>How many speakers are enrolled.</summary>
    public int Count
    {
        get
        {
            lock (_gate)
            {
                return _profiles.Count;
            }
        }
    }

    /// <summary>Enrolled speakers, ordered by name.</summary>
    public IReadOnlyList<SpeakerProfile> List()
    {
        lock (_gate)
        {
            List<SpeakerProfile> all = new List<SpeakerProfile>(_profiles.Values);
            all.Sort((left, right) => string.Compare(left.Name, right.Name, StringComparison.OrdinalIgnoreCase));
            return all;
        }
    }

    /// <summary>Looks a speaker up by name, case-insensitively.</summary>
    public bool TryGet(string name, out SpeakerProfile? profile)
    {
        lock (_gate)
        {
            return _profiles.TryGetValue(name ?? string.Empty, out profile);
        }
    }

    /// <summary>Stores a speaker model built from <paramref name="embeddings"/>, replacing any profile of the same
    /// name, and writes it to disk before returning. Pass <paramref name="phrase"/> when the utterances were
    /// repetitions of the wake phrase — text-dependent enrollment is what makes verification at wake-phrase length
    /// usable, and recording which phrase it was keeps a later reader from scoring the profile on unrelated speech
    /// and wondering why it underperforms.</summary>
    public SpeakerProfile Enroll(string name, IReadOnlyList<float[]> embeddings, string? phrase = null)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("A speaker profile needs a name.", nameof(name));
        }
        ArgumentNullException.ThrowIfNull(embeddings);
        if (embeddings.Count == 0)
        {
            throw new ArgumentException("A speaker profile needs at least one enrollment embedding.", nameof(embeddings));
        }

        string trimmed = name.Trim();
        float[][] normalized = new float[embeddings.Count][];
        for (int i = 0; i < embeddings.Count; i++)
        {
            if (embeddings[i] is null || embeddings[i].Length == 0)
            {
                throw new ArgumentException($"Enrollment embedding {i} is empty.", nameof(embeddings));
            }
            normalized[i] = SpeakerEmbeddingMath.Normalized(embeddings[i]);
        }

        SpeakerProfile profile = new SpeakerProfile
        {
            Name = trimmed,
            Centroid = SpeakerEmbeddingMath.Centroid(normalized),
            EnrollmentEmbeddings = normalized,
            Phrase = string.IsNullOrWhiteSpace(phrase) ? null : phrase.Trim(),
            EnrolledUtc = DateTimeOffset.UtcNow,
        };
        if (embeddings.Count < RecommendedEnrollmentUtterances)
        {
            Logs.Warning($"[Wake][Speaker] '{trimmed}' enrolled from {embeddings.Count} utterance(s); "
                + $"{RecommendedEnrollmentUtterances}-5 is the convention — fewer leaves the centroid dominated by one recording's room and mic.");
        }

        lock (_gate)
        {
            string stem = ResolveStem(trimmed);
            Write(stem, profile);
            _profiles[trimmed] = profile;
            _stems[trimmed] = stem;
        }
        Logs.Info($"[Wake][Speaker] Enrolled '{trimmed}' from {profile.UtteranceCount} utterance(s), "
            + $"{profile.Dimension}-d{(profile.IsTextDependent ? $", text-dependent on \"{profile.Phrase}\"" : string.Empty)}.");
        return profile;
    }

    /// <summary>Deletes a speaker and both of its files. Returns false when no such speaker was enrolled.</summary>
    public bool Remove(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            return false;
        }
        string trimmed = name.Trim();
        lock (_gate)
        {
            if (!_profiles.Remove(trimmed))
            {
                return false;
            }
            _stems.Remove(trimmed, out string? stem);
            if (stem is not null)
            {
                // Both files must go: a surviving sidecar resurrects the speaker on the next construction.
                Delete(Path.Combine(_root, stem + JsonExtension));
                Delete(Path.Combine(_root, stem + EmbeddingExtension));
            }
        }
        Logs.Info($"[Wake][Speaker] Removed '{trimmed}'.");
        return true;
    }

    /// <summary>Re-reads every profile from disk, discarding in-memory state. For a host that lets an operator drop
    /// sidecars in by hand.</summary>
    public void Reload()
    {
        lock (_gate)
        {
            _profiles.Clear();
            _stems.Clear();
            if (!System.IO.Directory.Exists(_root))
            {
                return;
            }
            foreach (string path in System.IO.Directory.EnumerateFiles(_root, "*" + JsonExtension, SearchOption.TopDirectoryOnly))
            {
                try
                {
                    SpeakerProfile profile = Read(path);
                    _profiles[profile.Name] = profile;
                    _stems[profile.Name] = Path.GetFileNameWithoutExtension(path);
                }
                catch (Exception ex)
                {
                    Logs.Error($"[Wake][Speaker] Skipping unreadable speaker profile '{path}'", ex);
                }
            }
        }
        if (_profiles.Count > 0)
        {
            Logs.Info($"[Wake][Speaker] Loaded {_profiles.Count} speaker profile(s) from '{_root}'.");
        }
    }

    /// <summary>Nearest enrolled centroid by cosine similarity, accepted only above <see cref="MatchThreshold"/>.
    /// The embedding need not be normalized. See <see cref="SpeakerMatch"/> — the nearest name and its score come
    /// back on a rejection too.</summary>
    public SpeakerMatch Identify(ReadOnlySpan<float> embedding) => Identify(embedding, MatchThreshold);

    /// <summary>As <see cref="Identify(ReadOnlySpan{float})"/> but with a per-call threshold, for calibration sweeps
    /// and for a wake word that wants to be stricter than the household default.</summary>
    public SpeakerMatch Identify(ReadOnlySpan<float> embedding, float threshold)
    {
        if (embedding.Length == 0)
        {
            throw new ArgumentException("Cannot identify from an empty embedding.", nameof(embedding));
        }
        string? bestName = null;
        float bestScore = float.NegativeInfinity;
        lock (_gate)
        {
            foreach (SpeakerProfile profile in _profiles.Values)
            {
                if (profile.Dimension != embedding.Length)
                {
                    Logs.Warning($"[Wake][Speaker] '{profile.Name}' is {profile.Dimension}-d but the query is "
                        + $"{embedding.Length}-d — skipping it. Re-enroll that speaker with the current encoder.");
                    continue;
                }
                float score = SpeakerEmbeddingMath.CosineSimilarity(embedding, profile.Centroid);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestName = profile.Name;
                }
            }
        }
        if (bestName is null)
        {
            return new SpeakerMatch(null, 0f, SpeakerMatchOutcome.NoProfiles, threshold);
        }
        SpeakerMatchOutcome outcome = bestScore >= threshold ? SpeakerMatchOutcome.Identified : SpeakerMatchOutcome.Unknown;
        return new SpeakerMatch(bestName, bestScore, outcome, threshold);
    }

    /// <summary>The default location, <c>{models}/audio/wake/speakers</c>.</summary>
    public static string DefaultDirectory() => Path.Combine(RepoPaths.ModelsRoot(), "audio", "wake", "speakers");

    /// <summary>Writes the binary first and the JSON index second, each via a temp file and an atomic move, so a
    /// crash mid-enroll leaves either the old profile or a mismatch that <see cref="Read"/> rejects — never a
    /// profile whose centroid silently belongs to somebody else.</summary>
    private void Write(string stem, SpeakerProfile profile)
    {
        System.IO.Directory.CreateDirectory(_root);
        string embeddingName = stem + EmbeddingExtension;
        string embeddingPath = Path.Combine(_root, embeddingName);
        string jsonPath = Path.Combine(_root, stem + JsonExtension);
        string embeddingTemp = embeddingPath + ".tmp";
        string jsonTemp = jsonPath + ".tmp";
        try
        {
            using (FileStream stream = File.Create(embeddingTemp))
            using (BinaryWriter writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: false))
            {
                writer.Write(Encoding.ASCII.GetBytes(EmbeddingMagic));
                writer.Write(EmbeddingFormatVersion);
                writer.Write(profile.UtteranceCount);
                writer.Write(profile.Dimension);
                foreach (float[] embedding in profile.EnrollmentEmbeddings)
                {
                    foreach (float value in embedding)
                    {
                        writer.Write(value);
                    }
                }
            }
            SpeakerProfileDocument document = new SpeakerProfileDocument
            {
                Name = profile.Name,
                Phrase = profile.Phrase,
                EnrolledUtc = profile.EnrolledUtc,
                UtteranceCount = profile.UtteranceCount,
                Dimension = profile.Dimension,
                EmbeddingFile = embeddingName,
            };
            File.WriteAllText(jsonTemp, JsonSerializer.Serialize(document, SpeakerProfileJsonContext.Default.SpeakerProfileDocument));
            File.Move(embeddingTemp, embeddingPath, overwrite: true);
            File.Move(jsonTemp, jsonPath, overwrite: true);
        }
        catch (Exception ex)
        {
            Logs.Error($"[Wake][Speaker] Failed to persist '{profile.Name}' to '{_root}'", ex);
            Delete(embeddingTemp);
            Delete(jsonTemp);
            throw;
        }
    }

    private SpeakerProfile Read(string jsonPath)
    {
        SpeakerProfileDocument? document = JsonSerializer.Deserialize(
            File.ReadAllText(jsonPath), SpeakerProfileJsonContext.Default.SpeakerProfileDocument);
        if (document is null || string.IsNullOrWhiteSpace(document.Name))
        {
            throw new InvalidDataException($"'{jsonPath}' carries no speaker name.");
        }
        string embeddingName = string.IsNullOrWhiteSpace(document.EmbeddingFile)
            ? Path.GetFileNameWithoutExtension(jsonPath) + EmbeddingExtension
            : Path.GetFileName(document.EmbeddingFile);
        string embeddingPath = Path.Combine(_root, embeddingName);
        if (!File.Exists(embeddingPath))
        {
            throw new FileNotFoundException($"Speaker '{document.Name}' has no embedding file at '{embeddingPath}'.", embeddingPath);
        }

        using FileStream stream = File.OpenRead(embeddingPath);
        using BinaryReader reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: false);
        string magic = Encoding.ASCII.GetString(reader.ReadBytes(EmbeddingMagic.Length));
        if (magic != EmbeddingMagic)
        {
            throw new InvalidDataException($"'{embeddingPath}' is not a speaker embedding file (magic '{magic}').");
        }
        int version = reader.ReadInt32();
        if (version != EmbeddingFormatVersion)
        {
            throw new InvalidDataException($"'{embeddingPath}' is format version {version}, this build reads {EmbeddingFormatVersion}.");
        }
        int count = reader.ReadInt32();
        int dimension = reader.ReadInt32();
        if (count != document.UtteranceCount || dimension != document.Dimension)
        {
            throw new InvalidDataException($"'{embeddingPath}' holds {count}x{dimension} but '{jsonPath}' claims "
                + $"{document.UtteranceCount}x{document.Dimension} — the pair was written by different enrollments.");
        }
        if (count <= 0 || dimension <= 0)
        {
            throw new InvalidDataException($"'{embeddingPath}' declares an empty {count}x{dimension} embedding set.");
        }

        float[][] embeddings = new float[count][];
        for (int i = 0; i < count; i++)
        {
            float[] embedding = new float[dimension];
            for (int d = 0; d < dimension; d++)
            {
                embedding[d] = reader.ReadSingle();
            }
            embeddings[i] = embedding;
        }
        return new SpeakerProfile
        {
            Name = document.Name.Trim(),
            Centroid = SpeakerEmbeddingMath.Centroid(embeddings),
            EnrollmentEmbeddings = embeddings,
            Phrase = document.Phrase,
            EnrolledUtc = document.EnrolledUtc,
        };
    }

    /// <summary>File stem for a name: the one already in use if this speaker is being re-enrolled, else a sanitized
    /// slug uniquified against the stems other speakers hold (two names can sanitize to the same slug).</summary>
    private string ResolveStem(string name)
    {
        if (_stems.TryGetValue(name, out string? existing))
        {
            return existing;
        }
        string slug = Slugify(name);
        string candidate = slug;
        for (int suffix = 2; IsStemTaken(candidate); suffix++)
        {
            candidate = $"{slug}-{suffix}";
        }
        return candidate;
    }

    private bool IsStemTaken(string stem)
    {
        foreach (string used in _stems.Values)
        {
            if (string.Equals(used, stem, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }
        return File.Exists(Path.Combine(_root, stem + JsonExtension));
    }

    private static string Slugify(string name)
    {
        StringBuilder builder = new StringBuilder(name.Length);
        foreach (char character in name)
        {
            builder.Append(char.IsLetterOrDigit(character) ? char.ToLowerInvariant(character) : '_');
        }
        string slug = builder.ToString().Trim('_');
        return slug.Length == 0 ? "speaker" : slug;
    }

    private static void Delete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (Exception ex)
        {
            Logs.Warning($"[Wake][Speaker] Could not delete '{path}': {ex.Message}");
        }
    }
}
