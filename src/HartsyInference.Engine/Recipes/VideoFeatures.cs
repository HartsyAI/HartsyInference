namespace HartsyInference.Engine.Recipes;

/// <summary>The conditioning a video recipe declares it can actually apply — the video counterpart of
/// <see cref="ImageFeatures"/>.
/// <para>This exists because the video side had no gate at all while transports were already populating
/// <see cref="Requests.VideoRequest.InitImage"/>: a family that never read it produced a plausible text-to-video clip
/// and gave no indication the user's image had been discarded. A silent drop is worse than a refusal, because nothing
/// in the output says the request was not honoured.</para></summary>
[Flags]
public enum VideoFeatures
{
    /// <summary>Text-to-video only; every conditioning object is rejected.</summary>
    None = 0,

    /// <summary>A start-frame image the clip is generated from (image-to-video).</summary>
    InitImage = 1,

    /// <summary>A target last frame, generated toward from the start frame.</summary>
    EndFrame = 2,

    /// <summary>LoRA stacks merged into the loaded weights.</summary>
    Lora = 4,
}
