namespace HartsyInference.PhoneGateway.Media;

/// <summary>Raw call recording. Off by default: recording calls needs the consent the law where you operate
/// requires (see <c>docs/Research/PHONE_GATEWAY.md</c>).</summary>
public sealed record RecordingOptions
{
    public bool Enabled { get; init; }

    /// <summary>Directory the per-call <c>*_in16k.pcm</c> and <c>*_out8k.pcm</c> files are written to.</summary>
    public string Directory { get; init; } = "";
}
