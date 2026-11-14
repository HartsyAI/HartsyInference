using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Infra;
using HartsyInference.Engine.Services;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Generates 3D output from an input: a mesh (GLB) or Gaussian-splat cloud (PLY) from an image with TripoSR, Hunyuan3D, TRELLIS, TRELLIS.2, Pixal3D or SAM 3D Objects (an RGBA image or --mask in, splats out; shape-only, or textured with --texture for TRELLIS.2 and Pixal3D), one GLB mesh per decomposed object from a scene mesh with CoLa3D, a rigged GLB from a mesh with UniRig, a textured GLB from a mesh and a reference image with Hunyuan3D Paint, a GLB with one coloured node per part from a mesh with Hunyuan3D-Part (P3-SAM), a GLB with every part regenerated as a complete shape with X-Part, a GLB from an image and a control signal (a point cloud, voxels, a bounding box or a skeleton) with Hunyuan3D-Omni, or hair cards from an image with HairDeck.</summary>
public sealed class ThreeDCommand : Command<ThreeDCommand.Settings>
{
    /// <summary>Options for <c>hartsy 3d</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Path to the input: a PNG image, or a .obj/.glb mesh for cola3d, unirig, hunyuan3d-part, hunyuan3d-xpart and the paint models.</summary>
        [CommandArgument(0, "<input>")]
        [Description("Input file: a PNG image (triposr, hunyuan3d and its -fast, -turbo and -mini builds, hunyuan3d-omni, trellis, trellis2, pixal3d, sam3d-objects) or a .obj/.glb mesh (cola3d, unirig, hunyuan3d-part, hunyuan3d-xpart, hunyuan3d-paint, hunyuan3d-paint21; the paint models also need --image). trellis2 uses the PNG alpha channel as the cutout when it has one.")]
        public string Input { get; init; } = "";

        /// <summary>Model id: triposr, hunyuan3d (also hunyuan3d-fast, hunyuan3d-turbo, hunyuan3d-mini), trellis, trellis2, pixal3d, sam3d-objects, cola3d, unirig, hunyuan3d-paint, hunyuan3d-paint21, hunyuan3d-part, hunyuan3d-xpart, or hairdeck.</summary>
        [CommandOption("-m|--model")]
        [Description("Model id: triposr, hunyuan3d (also hunyuan3d-fast, hunyuan3d-turbo, hunyuan3d-mini), trellis, trellis2, pixal3d, sam3d-objects (image + mask in, Gaussian-splat PLY out, or a GLB mesh with --mesh), cola3d, unirig (mesh in, rigged GLB out), hunyuan3d-paint (2.0, albedo) or hunyuan3d-paint21 (2.1, PBR) (mesh and --image in, textured GLB out), hunyuan3d-part (P3-SAM: mesh in, GLB with one coloured node per part out, labels and boxes printed), hunyuan3d-xpart (X-Part: mesh in, GLB with every P3-SAM part generated again as a complete shape, one coloured node per part, boxes printed; --steps flow steps, default 50, --grid marching-cubes grid per part, default 256), hunyuan3d-omni (image in, a control signal with --control, GLB mesh out; --steps flow steps, default 50, --grid marching-cubes grid, default 384, guidance 4.5), or hairdeck.")]
        public string Model { get; init; } = "triposr";

        /// <summary>Reference image for the texturing models (hunyuan3d-paint, hunyuan3d-paint21).</summary>
        [CommandOption("--image")]
        [Description("Reference image (PNG) the texturing models paint the input mesh from (hunyuan3d-paint, hunyuan3d-paint21). With --grid it sets the texture size (default 2048) and --steps the sampler steps (default 30 for hunyuan3d-paint, 15 for hunyuan3d-paint21).")]
        public string? Image { get; init; }

        /// <summary>hunyuan3d-omni: the control kind.</summary>
        [CommandOption("--control")]
        [Description("hunyuan3d-omni: the control signal beside the image: point (--control-file: obj, glb or a ply point cloud; its vertices are the points), voxel (--control-file: an obj or glb mesh with faces), bbox (--control-values: length,height,width, each in (0, 1]) or pose (--control-file: a bone file, or --control-values: six numbers per bone).")]
        public string? Control { get; init; }

        /// <summary>hunyuan3d-omni: geometry or bone file of the control.</summary>
        [CommandOption("--control-file")]
        [Description("hunyuan3d-omni: the file of a point, voxel or pose control (point: .obj, .glb or .ply; voxel: .obj or .glb; pose: a text file with six numbers per line).")]
        public string? ControlFile { get; init; }

        /// <summary>hunyuan3d-omni: the numbers of a bbox or pose control.</summary>
        [CommandOption("--control-values")]
        [Description("hunyuan3d-omni: the numbers of a bbox control (length,height,width) or a pose control (six numbers per bone), comma or space separated.")]
        public string? ControlValues { get; init; }

        /// <summary>Object mask for sam3d-objects.</summary>
        [CommandOption("--mask")]
        [Description("Object mask PNG for sam3d-objects, which lifts the masked object: the red channel is the coverage (non-zero is the object), as for --prompt-mask. Without it the input PNG alpha channel is the mask. It must have the input image size.")]
        public string? Mask { get; init; }

        /// <summary>Samples to generate (HairDeck only; this build accepts 1).</summary>
        [CommandOption("-n|--samples")]
        [Description("Samples to generate (hairdeck only; this build accepts 1).")]
        public int Samples { get; init; } = 1;

        /// <summary>Rigged output: skinned to the spring chains, with head-motion clips (HairDeck only; needs chains).</summary>
        [CommandOption("--rigged")]
        [Description("Rigged output with head-motion clips (hairdeck only; needs chains).")]
        public bool Rigged { get; init; }

        /// <summary>Skip the colour model (HairDeck only).</summary>
        [CommandOption("--no-colors")]
        [Description("Skip the colour model (hairdeck only).")]
        public bool NoColors { get; init; }

        /// <summary>Skip the spring-bone chains (HairDeck only).</summary>
        [CommandOption("--no-chains")]
        [Description("Skip the spring-bone chains (hairdeck only).")]
        public bool NoChains { get; init; }

        /// <summary>SAM 3D Objects: return a triangle mesh (GLB) instead of Gaussian splats (PLY).</summary>
        [CommandOption("--mesh")]
        [Description("sam3d-objects: return a triangle mesh (geometry-only GLB) instead of Gaussian splats; needs slat_decoder_mesh.safetensors beside the other checkpoints.")]
        public bool Mesh { get; init; }

        /// <summary>Texture the mesh with baked PBR textures (TRELLIS.2 and Pixal3D).</summary>
        [CommandOption("--texture")]
        [Description("Texture the mesh with baked PBR textures (trellis2 and pixal3d; needs the texture checkpoints, see docs/Research/TRELLIS2_ARCHITECTURE.md and docs/Research/PIXAL3D_ARCHITECTURE.md).")]
        public bool Texture { get; init; }

        /// <summary>Baked texture edge in texels (TRELLIS.2 and Pixal3D with --texture).</summary>
        [CommandOption("--texture-size")]
        [Description("Baked texture size in texels, 16 to 8192 (trellis2 and pixal3d with --texture; 0 = 2048).")]
        public int TextureSize { get; init; }

        /// <summary>A named bundle of quality settings.</summary>
        [CommandOption("--quality")]
        [Description("Quality preset: default (the model's own settings), high (the best that fits a 12 GB card in under ten minutes) or max (the best the model can give, whatever the time). Fills --steps, --grid and --texture-size where you leave them at 0; models without a preset reject it.")]
        public string? Quality { get; init; }

        /// <summary>Path to the model directory or checkpoint.</summary>
        [CommandOption("--model-path")]
        [Description("Path to the model directory or checkpoint.")]
        public string? ModelPath { get; init; }

        /// <summary>Compute backend selector.</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend: auto, cpu, cuda, or vulkan.")]
        public string Backend { get; init; } = "auto";

        /// <summary>Marching-cubes grid resolution (0 = model default).</summary>
        [CommandOption("--grid")]
        [Description("Grid resolution (0 = model default; cola3d: dual-grid voxels per axis, a multiple of 16, at most 1024, default 256, or 1024 with --prompt-point; trellis2: final dual-grid resolution, 512, 1024 or 1536, default 1024; 512 is one flow, 1024 and 1536 run the 512-to-1024 cascade; pixal3d: final dual-grid resolution, 1024 or 1536, default 1024, always the cascade).")]
        public int Grid { get; init; }

        /// <summary>Mask PNGs for cola3d, one per object (repeatable); all the same size, red channel is the coverage.</summary>
        [CommandOption("--prompt-mask")]
        [Description(
            "cola3d: a mask PNG per object, all the same size (repeatable). Red channel is the coverage. The view must match "
            + "the checkpoint's training view, which is not verified yet. Excludes --prompt-point.")]
        public string[]? PromptMasks { get; init; }

        /// <summary>Scene points for cola3d, one per object as x,y,z in scene units (repeatable).</summary>
        [CommandOption("--prompt-point")]
        [Description("cola3d: a point per object as x,y,z in the scene mesh's units (repeatable). Runs on the 1024 grid unless --grid is set, which is slow on CPU. Excludes --prompt-mask.")]
        public string[]? PromptPoints { get; init; }

        /// <summary>Denoising steps (Hunyuan3D, TRELLIS, TRELLIS.2, Pixal3D and HairDeck; 0 = default; trellis2 and pixal3d: 12).</summary>
        [CommandOption("--steps")]
        [Description("Denoising steps (Hunyuan3D, TRELLIS, TRELLIS.2, Pixal3D and HairDeck; 0 = default; trellis2 and pixal3d: 12).")]
        public int Steps { get; init; }

        /// <summary>pixal3d: horizontal field of view in radians.</summary>
        [CommandOption("--fov")]
        [Description("pixal3d: the photo's horizontal field of view in radians, in (0, pi/2); default 0.8576. Upstream estimates it with MoGe, which is not ported, so set it for a photo with strong perspective (upstream suggests 0.2 for a distorted result).")]
        public double? Fov { get; init; }

        /// <summary>RNG seed; &lt; 0 randomizes.</summary>
        [CommandOption("--seed")]
        [Description("RNG seed; negative randomizes.")]
        public int Seed { get; init; } = -1;

        /// <summary>Directory to save the GLB to.</summary>
        [CommandOption("-o|--output")]
        [Description("Directory to save the GLB (defaults to the output root).")]
        public string? Output { get; init; }

        /// <summary>Suppress progress output.</summary>
        [CommandOption("-q|--quiet")]
        [Description("Suppress progress output.")]
        public bool Quiet { get; init; }
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        if (!CommandRunner.RequireNonEmpty(settings.Input, "An input path is required (an image, or a mesh for cola3d and unirig).", out int exitCode))
            return exitCode;

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Mesh);
        if (spec.LocalPath is null)
        {
            // A catalog entry with files can be fetched with `hartsy pull <id>` (the 3D commands do not prompt for a download); a model with none (CoLa3D, UniRig) can only be pointed at a folder.
            string hint = spec.Catalog is { Assets.Count: > 0 } entry ? $", or download its catalog assets with [/][#2ea5e0]hartsy pull {Markup.Escape(entry.Id)}[/][red]" : string.Empty;
            AnsiConsole.MarkupLine($"[red]A model is required via[/] [#2ea5e0]--model-path[/][red]{hint}.[/]");
            return 1;
        }

        if (settings.PromptMasks is { Length: > 0 } && settings.PromptPoints is { Length: > 0 })
        {
            AnsiConsole.MarkupLine("[red]Give[/] [#2ea5e0]--prompt-mask[/] [red]or[/] [#2ea5e0]--prompt-point[/][red], not both.[/]");
            return 1;
        }

        ParamState parameters = new ParamState(Modality.Mesh) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        if (settings.PromptMasks is { Length: > 0 })
            parameters.Put("prompt-mask", string.Join('\n', settings.PromptMasks));
        if (settings.PromptPoints is { Length: > 0 })
            parameters.Put("prompt-point", string.Join('\n', settings.PromptPoints));
        if (settings.Image is not null)
            parameters.Put("image", settings.Image);
        if (settings.Mask is not null)
            parameters.Put("mask", settings.Mask);
        if (settings.Control is not null)
            parameters.Put("control", settings.Control);
        if (settings.ControlFile is not null)
            parameters.Put("control-file", settings.ControlFile);
        if (settings.ControlValues is not null)
            parameters.Put("control-values", settings.ControlValues);
        parameters.Put("grid", settings.Grid.ToString(CultureInfo.InvariantCulture));
        parameters.Put("steps", settings.Steps.ToString(CultureInfo.InvariantCulture));
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));
        if (settings.Fov is { } fov)
            parameters.Put("fov", fov.ToString("R", CultureInfo.InvariantCulture));
        parameters.Put("samples", settings.Samples.ToString(CultureInfo.InvariantCulture));
        parameters.Put("rigged", settings.Rigged ? "true" : "false");
        parameters.Put("no-colors", settings.NoColors ? "true" : "false");
        parameters.Put("no-chains", settings.NoChains ? "true" : "false");
        parameters.Put("texture", settings.Texture ? "true" : "false");
        parameters.Put("mesh", settings.Mesh ? "true" : "false");
        parameters.Put("texture-size", settings.TextureSize.ToString(CultureInfo.InvariantCulture));
        if (settings.Quality is not null)
            parameters.Put("quality", settings.Quality);
        string label = CommandRunner.ResolveLabel(spec, settings.Model);

        return CommandRunner.Run(Modality.Mesh, spec, settings.Input, parameters, settings.Backend, settings.Quiet,
            settings.Output, label, showResponseRule: false);
    }
}
