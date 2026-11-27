namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>One block's re-noise for the distilled solver: for each denoise step but the last, in step order, the noise of the whole video
/// and of the whole audio. Each array holds <c>steps - 1</c> copies of the block's noise layout, so a one-step run has empty arrays. The
/// caller supplies it, so a run can replay upstream's draws.</summary>
public sealed record DuoRenoise(float[] Video, float[] Audio);
