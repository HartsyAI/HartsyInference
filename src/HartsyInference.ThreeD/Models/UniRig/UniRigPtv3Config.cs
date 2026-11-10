using HartsyInference.Core.Exceptions;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>Hyper-parameters of the point transformer in the skin checkpoint (<c>ptv3obj</c> in <c>unirig_skin.yaml</c>).</summary>
public sealed record UniRigPtv3Config
{
    /// <summary>Input channels: coordinates, normal and three zero channels.</summary>
    public int InChannels { get; init; } = 9;

    public required int[] Depths { get; init; }
    public required int[] Channels { get; init; }
    public required int[] Heads { get; init; }

    /// <summary>Points per attention patch.</summary>
    public int PatchSize { get; init; } = 1024;

    /// <summary>Metres per grid cell of the voxelization the convolutions and the serialization run on.</summary>
    public float GridSize { get; init; } = 0.005f;

    /// <summary>Epsilon of the stem's BatchNorm (a running-statistics norm, so a fixed affine at inference).</summary>
    public float BatchNormEps { get; init; } = 1e-3f;

    /// <summary>Curves in checkpoint names; block <c>i</c> of a stage uses curve <c>i % length</c>. Upstream shuffles this list at random on every forward; this keeps the written order.</summary>
    public string[] Order { get; init; } = ["z", "z-trans", "hilbert", "hilbert-trans"];

    /// <summary>The shipped skin encoder: stages of 3, 3, 3, 6 and 16 blocks at widths 32 to 384.</summary>
    public static UniRigPtv3Config Default => new()
    {
        Depths = [3, 3, 3, 6, 16],
        Channels = [32, 64, 128, 256, 384],
        Heads = [2, 4, 8, 16, 24],
    };

    /// <summary>Throws for a config the encoder cannot build.</summary>
    public void Validate()
    {
        int stages = Depths?.Length ?? 0;
        if (stages == 0 || Channels?.Length != stages || Heads?.Length != stages || InChannels <= 0 || PatchSize <= 0 || GridSize <= 0f || Order is not { Length: > 0 })
        {
            throw new HartsyInferenceException("UniRig point transformer config is inconsistent.");
        }

        for (int s = 0; s < stages; s++)
        {
            if (Depths![s] < 0 || Channels![s] <= 0 || Heads![s] <= 0 || Channels[s] % Heads[s] != 0)
            {
                throw new HartsyInferenceException($"UniRig point transformer stage {s}: channels {Channels[s]} must divide by heads {Heads[s]}.");
            }
        }
    }
}
