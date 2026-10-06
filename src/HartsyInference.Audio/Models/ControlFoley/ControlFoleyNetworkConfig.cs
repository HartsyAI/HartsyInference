namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Hyper-parameters of the ControlFoley <c>AudioGenerationNetwork</c> (official <c>audio_model.py</c>).</summary>
public sealed record ControlFoleyNetworkConfig
{
    /// <summary>The released <c>large_44k</c> network (V1 projections, 54 blocks of which 36 are fused).</summary>
    public static ControlFoleyNetworkConfig Large44k { get; } = new()
    {
        LatentDim = 40,
        ClipDim = 1024,
        VisualDim = 768,
        SyncDim = 768,
        TextDim = 1024,
        AudioDim = 512,
        TimbreDim = 1536,
        HiddenDim = 64 * 14,
        Depth = 54,
        FusedDepth = 36,
        NumHeads = 14,
        LatentSeqLen = 345,
        ClipSeqLen = 64,
        VisualSeqLen = 32,
        SyncSeqLen = 192,
    };

    /// <summary>V2 projections (SiLU, frequency-sized timestep embedding) instead of V1 (SELU).</summary>
    public bool V2 { get; init; }

    public required int LatentDim { get; init; }
    public required int ClipDim { get; init; }
    public required int VisualDim { get; init; }
    public required int SyncDim { get; init; }
    public required int TextDim { get; init; }
    public required int AudioDim { get; init; }
    public required int TimbreDim { get; init; }
    public required int HiddenDim { get; init; }

    /// <summary>Total block count (joint + fused).</summary>
    public required int Depth { get; init; }

    /// <summary>Trailing blocks that only process the latent stream.</summary>
    public required int FusedDepth { get; init; }

    public required int NumHeads { get; init; }
    public float MlpRatio { get; init; } = 4f;
    public required int LatentSeqLen { get; init; }
    public required int ClipSeqLen { get; init; }
    public required int VisualSeqLen { get; init; }
    public required int SyncSeqLen { get; init; }
    public int TextSeqLen { get; init; } = 77;
    public int AudioSeqLen { get; init; } = 1;
    public int TimbreSeqLen { get; init; } = 1;

    /// <summary>Blocks that attend jointly over latent, audio, CLIP and text tokens.</summary>
    public int JointDepth => Depth - FusedDepth;

    /// <summary>The same network resized for another clip duration (the official <c>update_seq_lengths</c>).</summary>
    public ControlFoleyNetworkConfig WithSequenceLengths(ControlFoleyTemporalConfig temporal) => this with
    {
        LatentSeqLen = temporal.LatentSequenceLength,
        ClipSeqLen = temporal.ClipSequenceLength,
        VisualSeqLen = temporal.VisualSequenceLength,
        SyncSeqLen = temporal.SyncSequenceLength,
    };

    internal void Validate()
    {
        if (Depth <= FusedDepth || FusedDepth < 0)
        {
            throw new ArgumentException($"Depth ({Depth}) must exceed FusedDepth ({FusedDepth}).");
        }

        if (NumHeads <= 0 || HiddenDim % NumHeads != 0 || (HiddenDim / NumHeads) % 2 != 0 || HiddenDim % 2 != 0)
        {
            throw new ArgumentException("HiddenDim must split into NumHeads even head sizes.");
        }

        if (MlpRatio <= 0 || SyncSeqLen % 8 != 0)
        {
            throw new ArgumentException("MlpRatio must be positive and SyncSeqLen a multiple of 8.");
        }

        int[] positive =
        [
            LatentDim, ClipDim, VisualDim, SyncDim, TextDim, AudioDim, TimbreDim, LatentSeqLen, ClipSeqLen, VisualSeqLen,
            SyncSeqLen, TextSeqLen, AudioSeqLen, TimbreSeqLen,
        ];
        if (positive.Any(v => v <= 0))
        {
            throw new ArgumentException("All dimensions and sequence lengths must be positive.");
        }
    }
}
