using HartsyInference.Engine.Audio.Wake.Speakers;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>Speaker enrollment and the open-set decision — the parts that fail quietly.
///
/// <para>A broken centroid or a broken cosine does not crash: it silently answers with the wrong household member,
/// or waves a stranger through. A store that fails to round-trip through disk looks fine for the whole session it
/// was enrolled in and only forgets on the next restart, and a <c>Remove</c> that leaves one of the two sidecars
/// behind resurrects the speaker at the same moment. There is deliberately no end-to-end "speaker ID works" test:
/// a CAM++ that stops embedding is loud, and the weights are not on every machine.</para></summary>
public sealed class SpeakerProfileTests
{
    private const int Dimension = 192;

    [Fact]
    public void Centroid_NormalizesEachInput_SoAmplitudeDoesNotVote()
    {
        // The loud utterance is the same direction as the quiet one, scaled 50x. If the mean were taken over raw
        // vectors it would sit almost on top of the loud one; over unit vectors it is the true bisector.
        float[] quiet = Axis(0, 0.02f);
        float[] loud = Axis(1, 1.0f);

        float[] centroid = SpeakerEmbeddingMath.Centroid([quiet, loud]);

        Assert.Equal(centroid[0], centroid[1], 5);
    }

    [Fact]
    public void CosineSimilarity_IsOneForIdenticalVectors_AndScaleInvariant()
    {
        float[] vector = Ramp(seed: 7);
        float[] scaled = new float[Dimension];
        for (int i = 0; i < Dimension; i++)
        {
            scaled[i] = vector[i] * 3.5f;
        }

        Assert.Equal(1f, SpeakerEmbeddingMath.CosineSimilarity(vector, vector), 5);
        Assert.Equal(1f, SpeakerEmbeddingMath.CosineSimilarity(vector, scaled), 5);
        Assert.Equal(0f, SpeakerEmbeddingMath.CosineSimilarity(Axis(0, 1f), Axis(1, 1f)), 6);
        Assert.Equal(-1f, SpeakerEmbeddingMath.CosineSimilarity(Axis(0, 1f), Axis(0, -2f)), 5);
    }

    [Fact]
    public void Store_RoundTripsThroughDisk()
    {
        using TempDirectory directory = new TempDirectory();
        float[][] enrollments = [Ramp(1), Ramp(2), Ramp(3)];

        SpeakerProfile enrolled;
        {
            SpeakerProfileStore store = new SpeakerProfileStore(directory.Path);
            enrolled = store.Enroll("Kaleb Broo", enrollments, phrase: "hey hartsy");
        }

        SpeakerProfileStore reloaded = new SpeakerProfileStore(directory.Path);
        Assert.Equal(1, reloaded.Count);
        Assert.True(reloaded.TryGet("kaleb broo", out SpeakerProfile? restored));
        Assert.NotNull(restored);
        Assert.Equal("Kaleb Broo", restored!.Name);
        Assert.Equal("hey hartsy", restored.Phrase);
        Assert.Equal(3, restored.UtteranceCount);
        Assert.Equal(Dimension, restored.Dimension);
        for (int i = 0; i < Dimension; i++)
        {
            Assert.Equal(enrolled.Centroid[i], restored.Centroid[i], 6);
        }
        // A restored profile must be scored identically, not merely stored identically.
        Assert.Equal(1f, SpeakerEmbeddingMath.CosineSimilarity(enrolled.Centroid, restored.Centroid), 5);
    }

    [Fact]
    public void Store_Remove_DoesNotResurrectOnReload()
    {
        using TempDirectory directory = new TempDirectory();
        SpeakerProfileStore store = new SpeakerProfileStore(directory.Path);
        store.Enroll("Alice", [Ramp(1), Ramp(2), Ramp(3)]);
        store.Enroll("Bob", [Ramp(4), Ramp(5), Ramp(6)]);

        Assert.True(store.Remove("alice"));
        Assert.False(store.Remove("alice"));

        SpeakerProfileStore reloaded = new SpeakerProfileStore(directory.Path);
        Assert.Equal(1, reloaded.Count);
        Assert.False(reloaded.TryGet("Alice", out SpeakerProfile? _));
        Assert.True(reloaded.TryGet("Bob", out SpeakerProfile? _));
    }

    [Fact]
    public void Identify_RejectsAStranger_RatherThanReturningTheNearestName()
    {
        using TempDirectory directory = new TempDirectory();
        SpeakerProfileStore store = new SpeakerProfileStore(directory.Path);
        store.Enroll("Alice", [Axis(0, 1f)]);
        store.Enroll("Bob", [Axis(1, 1f)]);

        // Orthogonal to both centroids: cosine 0, below any sane threshold.
        SpeakerMatch stranger = store.Identify(Axis(2, 1f));
        Assert.Equal(SpeakerMatchOutcome.Unknown, stranger.Outcome);
        Assert.False(stranger.IsIdentified);
        Assert.Null(stranger.IdentifiedName);
        // The nearest candidate still comes back, because rejected (name, score) pairs are the calibration data.
        Assert.NotNull(stranger.Name);
        Assert.True(stranger.Score < store.MatchThreshold);

        SpeakerMatch known = store.Identify(Axis(1, 4f));
        Assert.Equal(SpeakerMatchOutcome.Identified, known.Outcome);
        Assert.Equal("Bob", known.IdentifiedName);
        Assert.Equal(1f, known.Score, 5);
    }

    [Fact]
    public void Identify_WithNobodyEnrolled_ReportsNoProfiles()
    {
        using TempDirectory directory = new TempDirectory();
        SpeakerProfileStore store = new SpeakerProfileStore(directory.Path);

        SpeakerMatch match = store.Identify(Ramp(1));

        Assert.Equal(SpeakerMatchOutcome.NoProfiles, match.Outcome);
        Assert.Null(match.Name);
        // An unrestricted wake word must still fire for a guest; a restricted one must not.
        Assert.True(match.Satisfies(null));
        Assert.False(match.Satisfies("Alice"));
    }

    private static float[] Axis(int index, float magnitude)
    {
        float[] vector = new float[Dimension];
        vector[index] = magnitude;
        return vector;
    }

    private static float[] Ramp(int seed)
    {
        float[] vector = new float[Dimension];
        for (int i = 0; i < Dimension; i++)
        {
            vector[i] = MathF.Sin((i + 1) * 0.37f * seed) + 0.1f * seed;
        }
        return vector;
    }

    private sealed class TempDirectory : IDisposable
    {
        public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "hartsy-speakers-" + Guid.NewGuid().ToString("N"));

        public TempDirectory() => Directory.CreateDirectory(Path);

        public void Dispose()
        {
            try
            {
                if (Directory.Exists(Path))
                {
                    Directory.Delete(Path, recursive: true);
                }
            }
            catch (IOException)
            {
                // A leftover temp directory is not worth failing a test over.
            }
        }
    }
}
