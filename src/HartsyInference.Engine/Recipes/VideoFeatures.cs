namespace HartsyInference.Engine.Recipes;

/// <summary>The conditioning a video recipe declares it can actually apply — the video counterpart of <see cref="ImageFeatures"/>.</summary>
/// <remarks>This exists because the video side had no gate at all while transports were already populating
/// <see cref="Requests.VideoRequest.InitImage"/>: a family that never read it produced a plausible text-to-video clip
/// and gave no indication the user's image had been discarded. A silent drop is worse than a refusal, because nothing
/// in the output says the request was not honoured.</remarks>
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

    /// <summary>Reference identity images conditioning the clip (MiniMax-H3 ref2va).</summary>
    ReferenceImages = 8,

    /// <summary>Reference video clips (optionally soundtracked) conditioning the clip (MiniMax-H3 ref2va).</summary>
    ReferenceVideos = 16,

    /// <summary>Standalone reference audio clips conditioning the soundtrack (MiniMax-H3 ref2va).</summary>
    ReferenceAudios = 32,

    /// <summary>A driving motion video (or pose/face override clips) the output animation follows (Wan-Animate).</summary>
    DrivingVideo = 64,

    /// <summary>One or more signed-frame visual or audio guide entries.</summary>
    Guides = 128,

    /// <summary>Per-video-row continuous denoising mask.</summary>
    VideoDenoiseMask = 256,

    /// <summary>Per-audio-row continuous denoising mask.</summary>
    AudioDenoiseMask = 512,

    /// <summary>One or more preprocessed video ControlNet streams.</summary>
    VideoControlNet = 1024,

    /// <summary>ControlNet-Union inpainting with visibility and masked-source channels.</summary>
    VideoInpaint = 2048,
}
