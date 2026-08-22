namespace HartsyInference.Diffusion.Requests;

/// <summary>Request parameters for text/image-to-video generation; extends <see cref="TextToImageRequest"/> with video-only knobs (frame count, frame rate, flow-match shift), which resolve against the model's <see cref="GenerationDefaults"/> the same way <see cref="TextToImageRequest.Steps"/>/<see cref="TextToImageRequest.CfgScale"/> do for images.</summary>
public record VideoGenerationRequest : TextToImageRequest
{
    /// <summary>Number of output frames; <c>null</c> = the pipeline's model default (e.g. Wan 81, LTX 161, LTX-2 121, HunyuanVideo 129, Kandinsky-5 121). Must satisfy the model's frame-count constraint (commonly <c>(n-1) % temporal_compression == 0</c>).</summary>
    public int? NumFrames { get; init; }

    /// <summary>Output frame rate (fps); <c>null</c> = model default (Wan 2.2 TI2V-5B 24, most others 16/24, Lance 12). Used for export timing and, for some models, RoPE temporal normalization.</summary>
    public float? Fps { get; init; }

    /// <summary>Optional flow-match timestep-shift override; <c>null</c> = the model/config default (reference ties this to resolution, e.g. Wan 5.0 at 720p / 3.0 at 480p).</summary>
    public float? FlowShift { get; init; }

    /// <summary>Wan-Animate-2 only: store the driving cache in BF16 (<c>true</c>) or F32 (<c>false</c>); <c>null</c> = resolve at the decision point (env override, else the low-VRAM auto policy). The Animate-2 recipe resolves once per generation and relays the answer here so a chunked video cannot mix cache dtypes.</summary>
    public bool? Animate2Bf16DrivingCache { get; init; }
}
