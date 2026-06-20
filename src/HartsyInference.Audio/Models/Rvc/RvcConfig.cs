using HartsyInference.Audio.Models.Vits;

namespace HartsyInference.Audio.Models.Rvc;

/// <summary>Configuration for RVC v2 (`SynthesizerTrnMs768NSFsid`) voice conversion. HuBERT/ContentVec content
/// (768-d) + F0 drive a VITS-derived synthesizer: a content+pitch encoder → prior, the reused
/// <see cref="VitsFlow"/>, and an NSF-HiFi-GAN decoder (the reused <see cref="VitsHiFiGan"/> with NSF source
/// injection). See <c>docs/Research/RVC_ARCHITECTURE.md</c>.</summary>
public sealed record RvcConfig
{
    /// <summary>The VITS core (inter/hidden 192, gin 256, 4 flows, NSF HiFi-GAN [10,10,2,2] @ 40 kHz).</summary>
    public VitsConfig Core { get; init; } = new()
    {
        InterChannels = 192, HiddenChannels = 192, FilterChannels = 768, NumHeads = 2, NumEncoderLayers = 6,
        WindowSize = 10, GinChannels = 256, FlowFlows = 4, FlowLayers = 3, FlowKernelSize = 5,
        ResBlock = "1", ResBlockKernelSizes = [3, 7, 11], ResBlockDilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        UpsampleRates = [10, 10, 2, 2], UpsampleInitialChannel = 512, UpsampleKernelSizes = [16, 16, 4, 4],
        SampleRate = 40_000,
    };

    public int ContentDim { get; init; } = 768;     // ContentVec/HuBERT v2 (256 for v1)
    public int SpkEmbedDim { get; init; } = 109;
    public float F0Min { get; init; } = 50f;
    public float F0Max { get; init; } = 1_100f;
    public int PitchBins { get; init; } = 256;

    public static RvcConfig V2_40k => new();
    public static RvcConfig V2_48k => new()
    {
        Core = V2_40k.Core with { UpsampleRates = [12, 10, 2, 2], UpsampleKernelSizes = [24, 20, 4, 4], SampleRate = 48_000 },
    };
}
