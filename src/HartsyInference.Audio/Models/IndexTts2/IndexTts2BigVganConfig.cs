namespace HartsyInference.Audio.Models.IndexTts2;

/// <summary>Config for <see cref="IndexTts2BigVganGenerator"/> — the stock <c>nvidia/bigvgan_v2_22khz_80band_256x</c>
/// vocoder IndexTTS-2 uses unmodified (confirmed from its real <c>config.json</c>): a plain 80-band/22050 Hz
/// mel → waveform generator, unlike IndexTTS-1.5's custom-trained, speaker-conditioned 24 kHz variant
/// (<see cref="HartsyInference.Audio.Models.IndexTts.IndexTtsBigVganConfig"/>).</summary>
public sealed record IndexTts2BigVganConfig
{
    public required int InputChannels { get; init; }           // mel bands (80)
    public required int UpsampleInitialChannel { get; init; }  // 1536
    public required int[] UpsampleRates { get; init; }         // [4,4,2,2,2,2] — hop 256
    public required int[] UpsampleKernelSizes { get; init; }   // [8,8,4,4,4,4]
    public required int[] ResblockKernelSizes { get; init; }   // [3,7,11]
    public required int[][] ResblockDilations { get; init; }   // [[1,3,5],[1,3,5],[1,3,5]]

    /// <summary>The real checkpoint's config.json: <c>use_tanh_at_final: false</c>, <c>use_bias_at_final: false</c>
    /// (the final <c>conv_post</c> has no bias term at all — confirmed absent from the real checkpoint's own
    /// keys) — both differ from IndexTTS-1.5's BigVGAN, which tanh's its output and biases <c>conv_post</c>.</summary>
    public static IndexTts2BigVganConfig Nvidia22kHz80Band => new()
    {
        InputChannels = 80,
        UpsampleInitialChannel = 1_536,
        UpsampleRates = [4, 4, 2, 2, 2, 2],
        UpsampleKernelSizes = [8, 8, 4, 4, 4, 4],
        ResblockKernelSizes = [3, 7, 11],
        ResblockDilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    };
}
