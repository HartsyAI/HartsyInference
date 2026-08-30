using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Generates a video (frame sequence) from a prompt with any registered video family. CUDA-only.</summary>
/// <remarks>Validation-pending per family — see <c>docs/Checklists/MODEL_STATUS_VIDEO.md</c>.</remarks>
public sealed class VideoCommand : Command<VideoCommand.Settings>
{
    /// <summary>Options for <c>hartsy video</c>.</summary>
    public sealed class Settings : PlacementCliSettings
    {
        /// <summary>The video description.</summary>
        [CommandArgument(0, "<prompt>")]
        [Description("The video description.")]
        public string Prompt { get; init; } = "";

        /// <summary>Model id (catalog) or path. Optional when <c>--model-path</c> is given.</summary>
        [CommandOption("-m|--model")]
        [Description("Catalog model id (ltx-video, wan, lance-video, ...) or a local path. Optional when --model-path is given.")]
        public string Model { get; init; } = "";

        /// <summary>Path to a video checkpoint in any registered family's layout.</summary>
        [CommandOption("--model-path")]
        [Description("Path to a checkpoint (file or folder) of any registered video family; pair it with -m <family> when the layout is ambiguous.")]
        public string? ModelPath { get; init; }

        /// <summary>Optional H3 checkpoint-profile id to confirm the header/hash detection.</summary>
        [CommandOption("--model-profile")]
        [Description("Confirm a detected H3 model profile. This cannot override incompatible tensors or hashes; --profile remains Engine tuning.")]
        public string? ModelProfile { get; init; }

        /// <summary>Compute backend (must be cuda for video).</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend (video requires cuda).")]
        public string Backend { get; init; } = "cuda";

        /// <summary>Negative prompt.</summary>
        [CommandOption("-n|--negative")]
        [Description("Negative prompt.")]
        public string Negative { get; init; } = "";

        /// <summary>Frame width in pixels; unset uses the family's native width.</summary>
        [CommandOption("--width")]
        [Description("Frame width in pixels (default: the model family's native width).")]
        public int? Width { get; init; }

        /// <summary>Frame height in pixels; unset uses the family's native height.</summary>
        [CommandOption("--height")]
        [Description("Frame height in pixels (default: the model family's native height).")]
        public int? Height { get; init; }

        /// <summary>Number of frames; unset uses the family's officially recommended count.</summary>
        [CommandOption("--frames")]
        [Description("Number of frames (default: the model family's recommended count).")]
        public int? Frames { get; init; }

        /// <summary>Denoising steps; unset uses the family's officially recommended count.</summary>
        [CommandOption("--steps")]
        [Description("Denoising steps (default: the model family's recommended count).")]
        public int? Steps { get; init; }

        /// <summary>Video flow-match shift.</summary>
        [CommandOption("--flow-shift")]
        [Description("Video flow-match shift; locked acceleration profiles reject incompatible values.")]
        public float? FlowShift { get; init; }

        /// <summary>Audio flow-match shift.</summary>
        [CommandOption("--audio-flow-shift")]
        [Description("Audio flow-match shift; MiniMax-H3 defaults independently from the video stream.")]
        public float? AudioFlowShift { get; init; }

        /// <summary>Sampler name.</summary>
        [CommandOption("--sampler")]
        [Description("Sampler name; locked H3 acceleration profiles require Euler.")]
        public string? Sampler { get; init; }

        /// <summary>Scheduler name.</summary>
        [CommandOption("--scheduler")]
        [Description("Sigma scheduler name; null keeps the detected profile recipe.")]
        public string? Scheduler { get; init; }

        /// <summary>Start frame for image-to-video; families without keyframe conditioning ignore it.</summary>
        [CommandOption("--init-image")]
        [Description("Path to an image the clip starts on (image-to-video).")]
        public string? InitImage { get; init; }

        /// <summary>End frame; supplying both anchors the clip at each end.</summary>
        [CommandOption("--end-frame")]
        [Description("Path to an image the clip ends on. Combine with --init-image for first-and-last-frame control.")]
        public string? EndFrame { get; init; }

        /// <summary>Driving motion video for character-animation families (Wan-Animate).</summary>
        [CommandOption("--driving-video")]
        [Description("Path to the driving motion video (Wan-Animate); its pose skeleton and face crop are auto-derived unless overridden.")]
        public string? DrivingVideo { get; init; }

        /// <summary>Pre-rendered pose/skeleton video overriding the pose branch's auto-preprocess.</summary>
        [CommandOption("--pose-video")]
        [Description("Path to a pre-rendered pose/skeleton video for the pose branch (overrides auto-preprocessing).")]
        public string? PoseVideo { get; init; }

        /// <summary>Pre-cropped face-square video overriding the face branch's auto-preprocess.</summary>
        [CommandOption("--face-video")]
        [Description("Path to a pre-cropped face-square video for the face branch (overrides auto-preprocessing).")]
        public string? FaceVideo { get; init; }

        /// <summary>Feed the raw driving clip to both branches instead of deriving pose skeleton + face crop.</summary>
        [CommandOption("--no-auto-preprocess")]
        [Description("Pass the raw driving video to the pose/face branches instead of auto-deriving the skeleton and face crop.")]
        public bool NoAutoPreprocess { get; init; }

        /// <summary>Second (low-noise) expert checkpoint for the Wan 2.2 A14B dual-expert pair.</summary>
        [CommandOption("--swap-model")]
        [Description("Path or model name of the Wan 2.2 low-noise expert; enables the dual-expert (MoE) schedule split.")]
        public string? SwapModel { get; init; }

        /// <summary>Fraction of steps given to the swap (low-noise) expert; unset uses the official boundary.</summary>
        [CommandOption("--swap-percent")]
        [Description("Fraction (0..1) of steps run by the swap model (warped through the flow shift); unset uses Wan 2.2's official boundary (0.875 T2V / 0.9 I2V).")]
        public double? SwapPercent { get; init; }

        /// <summary>Reference images for families that carry subject/style from references rather than pinning frames.</summary>
        [CommandOption("--ref-image")]
        [Description("Reference image to carry subject or style from; repeat for more (MiniMax-H3 takes up to 9).")]
        public string[]? ReferenceImages { get; init; }

        /// <summary>Reference video clips for families that carry subject/motion from reference footage.</summary>
        [CommandOption("--ref-video")]
        [Description("Reference video to carry subject or motion from; repeat for more (MiniMax-H3 takes up to 3).")]
        public string[]? ReferenceVideos { get; init; }

        /// <summary>Soundtracks paired by position to --ref-video.</summary>
        [CommandOption("--ref-video-audio")]
        [Description("Soundtrack (WAV) for the same-position --ref-video; repeat to pair more.")]
        public string[]? ReferenceVideoAudios { get; init; }

        /// <summary>Reference audio clips (WAV).</summary>
        [CommandOption("--ref-audio")]
        [Description("Reference audio clip (WAV) to condition on; repeat for more (MiniMax-H3 takes up to 3).")]
        public string[]? ReferenceAudios { get; init; }

        /// <summary>Arbitrary image guides expressed as FRAME=PATH.</summary>
        [CommandOption("--guide-image")]
        [Description("Image guide as FRAME=PATH; repeat. Negative frames resolve from the aligned target end.")]
        public string[]? GuideImages { get; init; }

        /// <summary>Arbitrary video guides expressed as FRAME=PATH.</summary>
        [CommandOption("--guide-video")]
        [Description("Video guide as FRAME=PATH; repeat. A frame cannot contain both an image and video guide.")]
        public string[]? GuideVideos { get; init; }

        /// <summary>Arbitrary audio guides expressed as FRAME=PATH.</summary>
        [CommandOption("--guide-audio")]
        [Description("Audio guide as FRAME=PATH; repeat. It merges with a visual guide at the same frame.")]
        public string[]? GuideAudios { get; init; }

        /// <summary>JSON guide manifest.</summary>
        [CommandOption("--guides-manifest")]
        [Description("JSON file containing arbitrary visual/audio H3 guides.")]
        public string? GuidesManifest { get; init; }

        /// <summary>Continuous video denoise mask.</summary>
        [CommandOption("--video-denoise-mask")]
        [Description("Image/video mask: white generates, black preserves; non-white regions require --video-mask-source.")]
        public string? VideoDenoiseMask { get; init; }

        /// <summary>Video source preserved by the denoise mask.</summary>
        [CommandOption("--video-mask-source")]
        [Description("Image/video source preserved by black video-mask regions.")]
        public string? VideoMaskSource { get; init; }

        /// <summary>Continuous audio denoise mask values.</summary>
        [CommandOption("--audio-denoise-mask")]
        [Description("JSON or delimited audio-mask values between 0 and 1; cadence defaults to 40 Hz.")]
        public string? AudioDenoiseMask { get; init; }

        /// <summary>Audio source preserved by the denoise mask.</summary>
        [CommandOption("--audio-mask-source")]
        [Description("Audio source preserved wherever the audio mask is below one.")]
        public string? AudioMaskSource { get; init; }

        /// <summary>Audio-mask sample cadence.</summary>
        [CommandOption("--audio-mask-rate")]
        [Description("Audio-mask values per second (default 40, the H3 audio-latent cadence).")]
        public float? AudioMaskRate { get; init; }

        /// <summary>One simple Fun ControlNet model.</summary>
        [CommandOption("--control-model")]
        [Description("MiniMax-H3 Fun ControlNet-Union checkpoint for the simple control slot.")]
        public string? ControlModel { get; init; }

        /// <summary>Preprocessed video for the simple control slot.</summary>
        [CommandOption("--control-video")]
        [Description("Already-preprocessed video for --control-model.")]
        public string? ControlVideo { get; init; }

        /// <summary>Simple control provenance kind.</summary>
        [CommandOption("--control-kind")]
        [Description("Control kind: canny, depth, hed, mlsd, pose, or custom. Inpaint requires --controls-manifest so its visibility and masked-source videos are explicit.")]
        public string? ControlKind { get; init; }

        /// <summary>Simple control strength.</summary>
        [CommandOption("--control-strength")]
        public double? ControlStrength { get; init; }

        /// <summary>Simple control start fraction.</summary>
        [CommandOption("--control-start")]
        public double? ControlStart { get; init; }

        /// <summary>Simple control end fraction.</summary>
        [CommandOption("--control-end")]
        public double? ControlEnd { get; init; }

        /// <summary>JSON control manifest for multiple streams.</summary>
        [CommandOption("--controls-manifest")]
        [Description("JSON file containing multiple independently windowed H3 controls.")]
        public string? ControlsManifest { get; init; }

        /// <summary>Video VAE override.</summary>
        [CommandOption("--video-vae")]
        [Description("Video VAE checkpoint override; legacy VAE remains the fallback.")]
        public string? VideoVae { get; init; }

        /// <summary>Audio VAE override.</summary>
        [CommandOption("--audio-vae")]
        [Description("Audio VAE checkpoint override, independent of the video VAE.")]
        public string? AudioVae { get; init; }

        /// <summary>Learned sparse-attention policy.</summary>
        [CommandOption("--sparse-attention")]
        [Description("Sparse-attention policy: auto, require, or disable. Known VSA profiles cannot silently run dense.")]
        public string? SparseAttention { get; init; }

        /// <summary>LoRAs to merge into the denoiser; families that don't declare LoRA support refuse them.</summary>
        [CommandOption("--lora")]
        [Description("LoRA to merge, by name or path; repeat for more. Merges into fp8 checkpoints too — the target weight is dequantized, merged, and requantized.")]
        public string[]? Loras { get; init; }

        /// <summary>Per-LoRA strengths, positionally matched to --lora; missing entries default to 1.0.</summary>
        [CommandOption("--lora-weight")]
        [Description("Strength for the same-position --lora (default 1.0).")]
        public double[]? LoraWeights { get; init; }

        /// <summary>Guidance scale; unset uses the family's officially recommended scale.</summary>
        [CommandOption("--cfg")]
        [Description("Guidance scale (default: the model family's recommended scale).")]
        public float? Cfg { get; init; }

        /// <summary>Frames per second; unset uses the family's native frame rate.</summary>
        [CommandOption("--fps")]
        [Description("Frame rate (default: the model family's native rate).")]
        public int? Fps { get; init; }

        /// <summary>RNG seed; &lt; 0 randomizes.</summary>
        [CommandOption("--seed")]
        [Description("RNG seed; negative randomizes.")]
        public int Seed { get; init; } = -1;

        /// <summary>Optional SeedVR2 restore pass over the generated frames.</summary>
        [CommandOption("--restore [MODEL]")]
        [Description("Restore/upscale the generated frames with SeedVR2 in the same run (default model seedvr2-3b). Generate small, restore up.")]
        public FlagValue<string> Restore { get; init; } = new();

        /// <summary>Restore target-area width (with --restore); default 1280.</summary>
        [CommandOption("--restore-width")]
        [Description("Restore target-area width (default 1280).")]
        public int? RestoreWidth { get; init; }

        /// <summary>Restore target-area height (with --restore); default 720.</summary>
        [CommandOption("--restore-height")]
        [Description("Restore target-area height (default 720).")]
        public int? RestoreHeight { get; init; }

        /// <summary>Directory to write the frame sequence into.</summary>
        [CommandOption("-o|--output")]
        [Description("Directory to write the frame sequence into (defaults to the output root).")]
        public string? Output { get; init; }

        /// <summary>Optional PNG path for the latest latent preview and its numbered temporal frames.</summary>
        [CommandOption("--preview-output")]
        [Description("Write live video-latent preview frames beside this PNG path as generation runs.")]
        public string? PreviewOutput { get; init; }

        /// <summary>Suppress progress output.</summary>
        [CommandOption("-q|--quiet")]
        [Description("Suppress progress output.")]
        public bool Quiet { get; init; }
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        if (!CommandRunner.RequireNonEmpty(settings.Prompt, "A prompt is required.", out int exitCode))
            return exitCode;

        if (!CommandRunner.RequireModelOrPath(settings.Model, settings.ModelPath, "--model", "--model-path", out exitCode))
            return exitCode;

        // Only flags the user actually passed are forwarded; anything omitted stays unset so the engine applies the
        // resolved family's official defaults instead of a generic guess.
        ParamState parameters = new ParamState(Modality.Video) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("negative", settings.Negative);
        parameters.PutIfSet("width", settings.Width);
        parameters.PutIfSet("height", settings.Height);
        parameters.PutIfSet("frames", settings.Frames);
        parameters.PutIfSet("steps", settings.Steps);
        parameters.PutIfSet("model-profile", settings.ModelProfile);
        parameters.PutIfSet("flow-shift", settings.FlowShift);
        parameters.PutIfSet("audio-flow-shift", settings.AudioFlowShift);
        parameters.PutIfSet("sampler", settings.Sampler);
        parameters.PutIfSet("scheduler", settings.Scheduler);
        parameters.PutIfSet("cfg", settings.Cfg);
        parameters.PutIfSet("fps", settings.Fps);
        parameters.PutIfSet("init-image", settings.InitImage);
        parameters.PutIfSet("end-frame", settings.EndFrame);
        parameters.PutIfSet("driving-video", settings.DrivingVideo);
        parameters.PutIfSet("pose-video", settings.PoseVideo);
        parameters.PutIfSet("face-video", settings.FaceVideo);
        if (settings.NoAutoPreprocess)
        {
            parameters.Put("no-auto-preprocess", "true");
        }
        parameters.PutIfSet("swap-model", settings.SwapModel);
        parameters.PutIfSet("swap-percent", settings.SwapPercent);
        parameters.PutIfSet("preview-output", settings.PreviewOutput);
        // Repeatable options collapse to one newline-joined value: the parameter bag is flat strings, and a path
        // cannot contain a newline.
        if (settings.ReferenceImages is { Length: > 0 })
        {
            parameters.Put("ref-images", string.Join('\n', settings.ReferenceImages));
        }
        if (settings.ReferenceVideos is { Length: > 0 })
        {
            parameters.Put("ref-videos", string.Join('\n', settings.ReferenceVideos));
        }
        if (settings.ReferenceVideoAudios is { Length: > 0 })
        {
            parameters.Put("ref-video-audios", string.Join('\n', settings.ReferenceVideoAudios));
        }
        if (settings.ReferenceAudios is { Length: > 0 })
        {
            parameters.Put("ref-audios", string.Join('\n', settings.ReferenceAudios));
        }
        if (settings.GuideImages is { Length: > 0 })
        {
            parameters.Put("guide-images", string.Join('\n', settings.GuideImages));
        }
        if (settings.GuideVideos is { Length: > 0 })
        {
            parameters.Put("guide-videos", string.Join('\n', settings.GuideVideos));
        }
        if (settings.GuideAudios is { Length: > 0 })
        {
            parameters.Put("guide-audios", string.Join('\n', settings.GuideAudios));
        }
        parameters.PutIfSet("guides-manifest", settings.GuidesManifest);
        parameters.PutIfSet("video-denoise-mask", settings.VideoDenoiseMask);
        parameters.PutIfSet("video-mask-source", settings.VideoMaskSource);
        parameters.PutIfSet("audio-denoise-mask", settings.AudioDenoiseMask);
        parameters.PutIfSet("audio-mask-source", settings.AudioMaskSource);
        parameters.PutIfSet("audio-mask-rate", settings.AudioMaskRate);
        parameters.PutIfSet("control-model", settings.ControlModel);
        parameters.PutIfSet("control-video", settings.ControlVideo);
        parameters.PutIfSet("control-kind", settings.ControlKind);
        parameters.PutIfSet("control-strength", settings.ControlStrength);
        parameters.PutIfSet("control-start", settings.ControlStart);
        parameters.PutIfSet("control-end", settings.ControlEnd);
        parameters.PutIfSet("controls-manifest", settings.ControlsManifest);
        parameters.PutIfSet("video-vae", settings.VideoVae);
        parameters.PutIfSet("audio-vae", settings.AudioVae);
        parameters.PutIfSet("sparse-attention", settings.SparseAttention);
        if (settings.Loras is { Length: > 0 })
        {
            parameters.Put("loras", string.Join('\n', settings.Loras));
            parameters.Put("lora-weights", string.Join('\n',
                settings.Loras.Select((_, i) => (settings.LoraWeights is not null && i < settings.LoraWeights.Length
                    ? settings.LoraWeights[i] : 1.0).ToString(CultureInfo.InvariantCulture))));
        }
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));
        if (settings.Restore.IsSet)
        {
            parameters.Put("restore", string.IsNullOrWhiteSpace(settings.Restore.Value) ? "seedvr2-3b" : settings.Restore.Value);
            parameters.PutIfSet("restore-width", settings.RestoreWidth);
            parameters.PutIfSet("restore-height", settings.RestoreHeight);
        }

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Video);
        if (!string.IsNullOrWhiteSpace(settings.ModelProfile))
        {
            spec = spec with { ProfileId = settings.ModelProfile };
        }
        string label = CommandRunner.ResolveLabel(spec, settings.Model, settings.ModelPath);

        (int? gpu, EngineOptions? engineOptions) = PlacementCli.Build(settings, settings.Backend,
            HartsyInference.Engine.Modality.Video, PlacementCli.TryModelBytes(settings.Model));
        return CommandRunner.Run(Modality.Video, spec, settings.Prompt, parameters, settings.Backend, settings.Quiet,
            settings.Output, label, showResponseRule: false, gpu, engineOptions);
    }
}
