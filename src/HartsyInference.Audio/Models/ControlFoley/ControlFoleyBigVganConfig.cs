namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Hyperparameters of the BigVGAN-v2 vocoder (<c>lib.bigvgan_v2.bigvgan.BigVGAN</c>, AMPBlock1 + SnakeBeta).</summary>
public sealed record ControlFoleyBigVganConfig
{
    public required int NumMels { get; init; }

    public required int UpsampleInitialChannel { get; init; }

    public required int[] UpsampleRates { get; init; }

    public required int[] UpsampleKernelSizes { get; init; }

    public required int[] ResblockKernelSizes { get; init; }

    public required int[][] ResblockDilations { get; init; }

    /// <summary>Final <c>tanh</c> when true, hard clamp to [-1, 1] otherwise.</summary>
    public bool UseTanhAtFinal { get; init; }

    /// <summary>Whether <c>conv_post</c> carries a bias.</summary>
    public bool UseBiasAtFinal { get; init; }

    /// <summary>Samples produced per mel frame.</summary>
    public int HopLength
    {
        get
        {
            int hop = 1;
            foreach (int rate in UpsampleRates) hop *= rate;
            return hop;
        }
    }

    /// <summary><c>nvidia/bigvgan_v2_44khz_128band_512x</c>, the vocoder ControlFoley loads for its 44k mode.</summary>
    public static ControlFoleyBigVganConfig V44k => new()
    {
        NumMels = 128,
        UpsampleInitialChannel = 1_536,
        UpsampleRates = [8, 4, 2, 2, 2, 2],
        UpsampleKernelSizes = [16, 8, 4, 4, 4, 4],
        ResblockKernelSizes = [3, 7, 11],
        ResblockDilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    };
}
