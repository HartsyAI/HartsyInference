namespace HartsyInference.Audio.Models.IndexTts;

/// <summary>Config for <see cref="IndexTtsBigVganGenerator"/> — IndexTTS-1.5's custom-trained 24 kHz BigVGAN-v2
/// (AMPBlock1 / SnakeBeta, "AMP" variant), verified against the real <c>bigvgan_generator.pth</c> checkpoint and
/// <c>config.yaml</c>'s <c>bigvgan:</c> section.</summary>
public sealed record IndexTtsBigVganConfig
{
    public required int GptDim { get; init; }                 // conv_pre input channels (1280)
    public required int SpeakerEmbeddingDim { get; init; }     // cond_layer/conds[i] input (512)
    public required int UpsampleInitialChannel { get; init; }  // 1536
    public required int[] UpsampleRates { get; init; }         // [4,4,4,4,2,2]
    public required int[] UpsampleKernelSizes { get; init; }   // [8,8,4,4,4,4]
    public required int[] ResblockKernelSizes { get; init; }   // [3,7,11]
    public required int[][] ResblockDilations { get; init; }   // [[1,3,5],[1,3,5],[1,3,5]]

    public static IndexTtsBigVganConfig V1_5 => new()
    {
        GptDim = 1_280,
        SpeakerEmbeddingDim = 512,
        UpsampleInitialChannel = 1_536,
        UpsampleRates = [4, 4, 4, 4, 2, 2],
        UpsampleKernelSizes = [8, 8, 4, 4, 4, 4],
        ResblockKernelSizes = [3, 7, 11],
        ResblockDilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    };
}
