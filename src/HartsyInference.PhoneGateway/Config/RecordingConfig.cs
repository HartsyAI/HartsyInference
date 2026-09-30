namespace HartsyInference.PhoneGateway.Config;

/// <summary>Raw call recording (<c>recording</c> section). Off by default: recording calls needs the consent the law
/// where you operate requires (see <c>docs/Research/PHONE_GATEWAY.md</c>).</summary>
public sealed record RecordingConfig
{
    public bool Enabled { get; set; }

    /// <summary>Directory the per-call <c>*_in16k.pcm</c> and <c>*_out8k.pcm</c> files are written to.</summary>
    public string Directory { get; set; } = "";
}
