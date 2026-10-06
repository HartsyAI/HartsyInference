namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Sequence lengths of every ControlFoley modality for one clip duration; port of the official
/// <c>TemporalConfiguration</c> (<c>temporal_config.py</c>).</summary>
public sealed record ControlFoleyTemporalConfig
{
    /// <summary>The released 44.1 kHz configuration (8 s clips).</summary>
    public static ControlFoleyTemporalConfig Default44k { get; } = new()
    {
        TotalTimeSeconds = 8.0,
        AudioSampleRate = 44100,
        SpecFrameFrequency = 512,
    };

    /// <summary>Clip duration in seconds.</summary>
    public required double TotalTimeSeconds { get; init; }

    /// <summary>Output sample rate in Hz.</summary>
    public required int AudioSampleRate { get; init; }

    /// <summary>Hop (samples) of one spectrogram frame.</summary>
    public required int SpecFrameFrequency { get; init; }

    /// <summary>Spectrogram frames per latent step.</summary>
    public int LatentReductionFactor { get; init; } = 2;

    /// <summary>CLIP frames per second.</summary>
    public int ClipFrameFrequency { get; init; } = 8;

    /// <summary>CAV-MAE frames per second.</summary>
    public int VisualFrameFrequency { get; init; } = 4;

    /// <summary>Synchformer input frames per second.</summary>
    public int SyncFrameFrequency { get; init; } = 25;

    /// <summary>Video frames per Synchformer segment.</summary>
    public int SyncSegmentFrameCount { get; init; } = 16;

    /// <summary>Video frames between Synchformer segment starts.</summary>
    public int SyncStrideFrames { get; init; } = 8;

    /// <summary>Temporal downsampling inside Synchformer.</summary>
    public int SyncDownsamplingFactor { get; init; } = 2;

    /// <summary>Latent sequence length: <c>ceil(seconds * sampleRate / (hop * reduction))</c>.</summary>
    public int LatentSequenceLength
    {
        get
        {
            Validate();
            double numerator = TotalTimeSeconds * AudioSampleRate;
            return (int)Math.Ceiling(numerator / (SpecFrameFrequency * LatentReductionFactor));
        }
    }

    /// <summary>Audio samples covered by a whole number of latent steps.</summary>
    public int TotalAudioSampleCount => LatentSequenceLength * SpecFrameFrequency * LatentReductionFactor;

    /// <summary>CAV-MAE feature sequence length (truncated, as <c>int()</c> in python).</summary>
    public int VisualSequenceLength
    {
        get
        {
            Validate();
            return (int)(TotalTimeSeconds * VisualFrameFrequency);
        }
    }

    /// <summary>CLIP feature sequence length (truncated).</summary>
    public int ClipSequenceLength
    {
        get
        {
            Validate();
            return (int)(TotalTimeSeconds * ClipFrameFrequency);
        }
    }

    /// <summary>Synchformer feature sequence length from overlapping segments after downsampling.</summary>
    public int SyncSequenceLength
    {
        get
        {
            Validate();
            double totalFrames = TotalTimeSeconds * SyncFrameFrequency;
            double segments = Math.Floor((totalFrames - SyncSegmentFrameCount) / SyncStrideFrames) + 1;
            if (segments < 0)
            {
                throw new InvalidOperationException($"Invalid segment count calculation: {segments}");
            }

            return (int)(segments * SyncSegmentFrameCount / SyncDownsamplingFactor);
        }
    }

    private void Validate()
    {
        if (TotalTimeSeconds <= 0 || AudioSampleRate <= 0 || SpecFrameFrequency <= 0 || LatentReductionFactor <= 0 ||
            ClipFrameFrequency <= 0 || VisualFrameFrequency <= 0 || SyncFrameFrequency <= 0 ||
            SyncSegmentFrameCount <= 0 || SyncStrideFrames <= 0 || SyncDownsamplingFactor <= 0)
        {
            throw new ArgumentException("Every temporal configuration value must be positive.");
        }
    }
}
