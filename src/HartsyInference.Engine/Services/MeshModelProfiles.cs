using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>The profiles of every described 3D model. The rules they state are the ones <see cref="MeshRequestValidator"/> enforces; a test walks the legal values and the first illegal ones of each field through the validator.</summary>
internal static class MeshModelProfiles
{
    private const string Hunyuan2Terms = "Tencent Hunyuan 3D 2.0 Community License";
    private const string SonataNonCommercial = "The Sonata backbone weights (facebook/sonata) are released under CC-BY-NC 4.0, restricted by the non-commercial terms of datasets such as HM3D and ArkitScenes (github.com/facebookresearch/sonata); whether the fine-tuned P3-SAM checkpoint inherits that is not settled by either licence text, so treat the Sonata-derived part as non-commercial unless the licensors say otherwise";

    private static readonly string[] HunyuanRestrictions =
    [
        "Not licensed in the European Union, the United Kingdom or South Korea (use or display of the Works or Output there)",
        "A product with more than 1 million monthly active users needs a separate Tencent licence",
        "The Works and their Output may not be used to improve another AI model",
    ];

    private static readonly Dictionary<string, MeshModelProfile> Profiles = Build();

    /// <summary>The profile of <paramref name="id"/>, or null when it has none.</summary>
    public static MeshModelProfile? Find(string id) => Profiles.GetValueOrDefault(id);

    private static Dictionary<string, MeshModelProfile> Build()
    {
        List<MeshModelProfile> all =
        [
            Profile("triposr", MeshInputKind.Image, MeshOutputKinds.Mesh,
                [Ignored(nameof(MeshRequest.Steps), "TripoSR is one feed-forward pass."), Grid(null, null, 1, null, "Marching-cubes samples per axis; memory grows with its cube."),
                    Rejected(nameof(MeshRequest.FieldOfViewRadians)), IgnoredSeed("There is no sampler to seed."),
                    .. NoHairDeck(), .. NoTexture()],
                new MeshLicence { Summary = "MIT (stabilityai/TripoSR)" },
                new MeshCost { Basis = "Verified on CPU and an RTX 3060; no time or memory figure is recorded." },
                note: "Expects the foreground cut out and composited on mid-gray."),
            .. HunyuanShapeProfiles(),
            Profile("trellis", MeshInputKind.Image, MeshOutputKinds.Splats,
                [Steps("25", 1, null, "Sampler steps of both flows."), Ignored(nameof(MeshRequest.GridResolution), "The output is a splat cloud."),
                    Rejected(nameof(MeshRequest.FieldOfViewRadians)), Seed(), .. NoHairDeck(), .. NoTexture()],
                new MeshLicence { Summary = "MIT (TRELLIS-image-large)" },
                new MeshCost { Basis = "No time or memory figure is recorded." },
                note: "Returns Gaussian splats (ply), not a mesh."),
            Profile(MeshModelIds.Trellis2, MeshInputKind.Image, MeshOutputKinds.Mesh | MeshOutputKinds.TexturedMesh,
                [Steps(null, 0, 200, "Sampler steps of each flow; 0 takes the checkpoint's schedule."), Grid("1024", ["512", "1024", "1536"], null, null, "Final dual-grid resolution."),
                    Rejected(nameof(MeshRequest.FieldOfViewRadians)), Seed(), .. NoHairDeck(), .. TextureOptional()],
                new MeshLicence { Summary = "MIT (TRELLIS.2, TRELLIS-image-large); DINOv3 under the DINOv3 License; BiRefNet-matting MIT" },
                new MeshCost { Basis = "Timings differ by path (512, 1024 cascade, textured); no whole-card peak is recorded for the default request." },
                note: "Needs the foreground cut out: an image with alpha, or the BiRefNet matting checkpoint installed."),
            Profile(MeshModelIds.Pixal3D, MeshInputKind.Image, MeshOutputKinds.Mesh | MeshOutputKinds.TexturedMesh,
                [Steps(null, 0, 200, "Sampler steps of each flow; 0 takes the checkpoint's schedule."), Grid("1024", ["1024", "1536"], null, null, "Final dual-grid resolution; there is no 512 pipeline."),
                    Used(nameof(MeshRequest.FieldOfViewRadians), "0.8575560450553894", 0, Math.PI / 2, "radians", "The photo's horizontal field of view, strictly between 0 and pi/2."),
                    Seed(), .. NoHairDeck(), .. TextureOptional()],
                new MeshLicence { Summary = "MIT (Pixal3D); DINOv3 under the DINOv3 License; the NAF weights' licence is not stated" },
                new MeshCost { PeakVramGiB = 9.3, TypicalSeconds = 386, Basis = "RTX 3060, 1024 cascade, shape-only, 12 steps per flow, one image (a bust); 1536 peaks at 11.6." },
                peak: request => request.GridResolution == 1536 ? 11.6 : 9.3,
                note: "naf_release.pth must be placed in the model folder by hand; needs the foreground cut out like TRELLIS.2."),
            Profile(MeshModelIds.Sam3dObjects, MeshInputKind.ImageWithMask, MeshOutputKinds.Splats | MeshOutputKinds.Mesh,
                [Steps("25", 0, null, "Steps of the structure flow and of the SLAT flow."), Rejected(nameof(MeshRequest.GridResolution)),
                    Rejected(nameof(MeshRequest.FieldOfViewRadians)), Seed(), .. NoHairDeck(), .. NoTexture()],
                new MeshLicence
                {
                    Summary = "SAM License (binds on use and redistribution)",
                    Restrictions = ["Use ends for trade-controlled parties and for military, nuclear, espionage and weapons end uses; read the licence text fetched beside the weights"],
                },
                new MeshCost { TypicalSeconds = 450, Basis = "Splats: RTX 3060, F32 flows, one example image; the SLAT flow fills the card, so no peak is stated. Mesh: RTX 3060, F16 flow weights (numerics.flowF16Weights), one bust, 340 s and an 11.8 GiB peak of the 12 GiB card (F32 flows are not recorded for the mesh path)." },
                peak: request => request.Mesh ? 11.8 : null,
                note: "The image's alpha plane is the mask of the object to lift."),
            Profile(MeshModelIds.HairDeck, MeshInputKind.Image, MeshOutputKinds.Mesh | MeshOutputKinds.RiggedMesh,
                [Steps("50", 1, null, "Euler steps of the card sampler."), Ignored(nameof(MeshRequest.GridResolution), "HairDeck has no grid."),
                    Rejected(nameof(MeshRequest.FieldOfViewRadians)),
                    Seed(),
                    new MeshFieldSpec { Name = nameof(MeshRequest.Samples), Access = MeshFieldAccess.Used, Default = "1", Values = ["1"], Note = "Only one sample per request." },
                    Used(nameof(MeshRequest.Colors), "true", null, null, null, "Colour the cards with the colour model."),
                    Used(nameof(MeshRequest.Chains), "true", null, null, null, "Also generate the spring-bone chains."),
                    Used(nameof(MeshRequest.Rigged), "false", null, null, null, "Rigged output, skinned to the chains with head-motion clips; needs chains."),
                    .. NoTexture()],
                new MeshLicence { Summary = "HairDeck weights Apache-2.0; SAM 3 under the SAM License (gated); DINOv3 License (gated); BiRefNet-matting MIT" },
                new MeshCost { Basis = "Run on one demo image on an RTX 3060; no time or memory figure is recorded." },
                note: "Mattes the image with SAM 3 and BiRefNet before sampling."),
            Profile("cola3d", MeshInputKind.SceneMesh, MeshOutputKinds.Mesh,
                [Ignored(nameof(MeshRequest.Steps), "The decomposer is one forward pass."), Grid("256", null, 16, 1024, "Dual-grid voxels per axis, a multiple of 16; a point prompt defaults to 1024."),
                    Rejected(nameof(MeshRequest.FieldOfViewRadians)), IgnoredSeed("The decomposer is deterministic."), .. NoHairDeck(), .. NoTexture()],
                new MeshLicence { Summary = "Not recorded in the engine's documents" },
                new MeshCost { Basis = "No time or memory figure is recorded." },
                note: "Takes a scene mesh and, optionally, a mask or point prompt in ScenePrompt."),
            Profile("unirig", MeshInputKind.SceneMesh, MeshOutputKinds.RiggedMesh,
                [Rejected(nameof(MeshRequest.Steps)), Rejected(nameof(MeshRequest.GridResolution)), Rejected(nameof(MeshRequest.FieldOfViewRadians)),
                    Seed(), .. NoHairDeck(), .. NoTexture()],
                new MeshLicence { Summary = "MIT (code and weights); training data CC-BY-4.0" },
                new MeshCost { Basis = "CPU F32 skinning of one mesh took 400 s; no GPU figure is recorded." },
                note: "Skeleton and skin weights for a triangle mesh; the result is a rigged glb."),
            Profile("hunyuan3d-part", MeshInputKind.SceneMesh, MeshOutputKinds.PartMesh,
                [Rejected(nameof(MeshRequest.Steps)), Rejected(nameof(MeshRequest.GridResolution)), Rejected(nameof(MeshRequest.FieldOfViewRadians)),
                    Seed("Seeds the surface sampling."), .. NoHairDeck(), .. NoTexture()],
                new MeshLicence
                {
                    Summary = "Tencent Hunyuan 3D-Part Community License; its Sonata backbone comes from facebook/sonata, whose weights are CC-BY-NC 4.0",
                    Restrictions = [.. HunyuanRestrictions, SonataNonCommercial],
                },
                new MeshCost { PeakVramGiB = 4.7, TypicalSeconds = 53, Basis = "RTX 3060 through the CLI on a 1.0 M-face bust, peak 4.7." },
                peak: _ => 4.7,
                note: "Segments a mesh into parts; the labels and boxes come back in MeshResult.Parts."),
            Profile("hunyuan3d-xpart", MeshInputKind.SceneMesh, MeshOutputKinds.PartMesh,
                [Steps("50", 0, 1000, "Euler steps of the part flow; 0 takes 50."),
                    Grid("256", null, 32, 512, "Marching-cubes grid per axis of each part, decoded densely (the host holds the cube: 0.5 GB at 512); 0 takes 256, upstream extracts on an octree at 512."),
                    Rejected(nameof(MeshRequest.FieldOfViewRadians)),
                    Seed("Fixes the surface samples, the part ids and the noise; a negative value draws a random seed."), .. NoHairDeck(), .. NoTexture()],
                new MeshLicence
                {
                    Summary = "Tencent Hunyuan 3D-Part Community License; its Sonata backbone comes from facebook/sonata, whose weights are CC-BY-NC 4.0",
                    Restrictions = [.. HunyuanRestrictions, SonataNonCommercial],
                },
                new MeshCost { PeakVramGiB = 9.5, TypicalSeconds = 385, Basis = "RTX 3060, a 40,080-face bust split into 3 parts, grid 256, 50 steps: P3-SAM 66 s, conditioning 21 s, flow 140 s, decode 141 s." },
                peak: _ => 9.5,
                note: "P3-SAM splits the mesh into parts and each part is generated again as a complete shape; the boxes and face counts come back in MeshResult.Parts. Decimate a dense mesh to about 40,000 faces first. The setting numerics.xpartFastDecode (off) decodes the parts with FlashVDM: about 113 s for three parts instead of 141 s, on a finer grid (504 cells), approximating the dense field."),
            Profile(MeshModelIds.Omni, MeshInputKind.ImageAndControl, MeshOutputKinds.Mesh,
                [Steps("50", 0, 1000, "Euler steps of the flow; 0 takes 50 (49 evaluations, each with and without the image)."),
                    Grid("384", null, 32, 512, "Marching-cubes grid per axis, decoded densely (the host holds the cube: 0.5 GB at 512); 0 takes 384, upstream's script uses 512."),
                    Rejected(nameof(MeshRequest.FieldOfViewRadians)), Seed(), .. NoHairDeck(), .. NoTexture()],
                new MeshLicence
                {
                    Summary = "Tencent Hunyuan 3D Omni Community License; DINOv2 is Apache-2.0",
                    Restrictions = [.. HunyuanRestrictions, "A product built on it must name its real provider and state that Tencent is not affiliated with, sponsoring or endorsing it"],
                },
                new MeshCost { PeakVramGiB = 9.1, TypicalSeconds = 394, Basis = "RTX 3060, a bust photo and a 2,048-point control, 50 steps, guidance 7.5, grid 384: load 27 s, flow 132 s, decode and marching cubes 235 s; grid 256 took 285 s and peaked at 8.9 GB." },
                peak: _ => 9.1,
                note: "The image guides appearance and the control guides shape: give the image's foreground on white (Preprocess Auto does it) and a control in the model's own frame. Guidance is 4.5, the value upstream's script uses."),
            Paint("hunyuan3d-paint", "Tencent Hunyuan 3D 2.0 Community License", "Albedo only; metallic is 0 and roughness 1.",
                new MeshCost { Basis = "A 30-step run was estimated at 175 s from a 12-step run on a card shared with another process; no isolated peak." }, peak: null),
            Paint("hunyuan3d-paint21", "Tencent Hunyuan 3D 2.1 Community License", "PBR texture (base color, metallic, roughness).",
                new MeshCost { PeakVramGiB = 9.8, TypicalSeconds = 99, Basis = "RTX 3060, 512 px views, 15 UniPC steps, no upscaler, a 19,800-triangle mesh; the first run on a saturated card took 246 s." },
                peak: _ => 9.8),
        ];
        return all.ToDictionary(profile => profile.Id, StringComparer.OrdinalIgnoreCase);
    }

    private static IEnumerable<MeshModelProfile> HunyuanShapeProfiles()
    {
        foreach (string id in MeshModelIds.Hunyuan3D)
        {
            yield return Profile(id, MeshInputKind.Image, MeshOutputKinds.Mesh,
                [Steps(null, 1, null, "Sampler steps; the build's own default (50 standard, 10 Fast, 5 Turbo) when unset."),
                    Grid(null, null, 1, null, "Marching-cubes samples per axis; the checkpoint's default when unset."),
                    Rejected(nameof(MeshRequest.FieldOfViewRadians)), Seed(), .. NoHairDeck(), .. NoTexture()],
                new MeshLicence { Summary = Hunyuan2Terms, Restrictions = HunyuanRestrictions },
                new MeshCost { Basis = "Verified stage by stage against the Python reference; the recorded times are historical and differ by build." },
                note: "Expects the foreground recentred on white at 512 px; the engine does not prepare the image yet.");
        }
    }

    private static MeshModelProfile Paint(string id, string terms, string note, MeshCost cost, Func<MeshRequest, double?>? peak) =>
        Profile(id, MeshInputKind.SceneMeshAndImage, MeshOutputKinds.TexturedMesh,
            [Steps(null, 0, null, "Sampler steps; 0 takes the checkpoint's default."),
                Used(nameof(MeshRequest.GridResolution), "2048", 0, null, "texels", "The baked texture's edge, not a grid; 0 takes 2048."),
                Rejected(nameof(MeshRequest.FieldOfViewRadians)), Seed("Seeds the multiview noise."),
                .. NoHairDeck(), .. NoTexture()],
            new MeshLicence
            {
                Summary = terms + "; the UNet is built on SD 2.1-base (CreativeML Open RAIL++-M)",
                Restrictions = HunyuanRestrictions,
            },
            cost, peak, note + " The mesh and the reference image are both required.");

    private static MeshModelProfile Profile(
        string id, MeshInputKind input, MeshOutputKinds outputs, IEnumerable<MeshFieldSpec> fields, MeshLicence licence, MeshCost cost,
        Func<MeshRequest, double?>? peak = null, string? note = null) =>
        new() { Id = id, Input = input, Outputs = outputs, Fields = [.. fields, MeshField(id), PreprocessField(id), ControlField(id), .. OutputFields(outputs)], Licence = licence, Cost = cost, PeakGiB = peak, Note = note };

    private static MeshFieldSpec[] OutputFields(MeshOutputKinds outputs)
    {
        if (outputs == MeshOutputKinds.Splats)
        {
            return
            [
                Ignored(nameof(MeshRequest.PreviewTriangles), "A splat result has no triangles to decimate."),
                Ignored(nameof(MeshRequest.Frame), "A splat result is returned in the object's own frame."),
            ];
        }
        return
        [
            Used(nameof(MeshRequest.PreviewTriangles), "0", 0, null, "triangles", "Also return a copy decimated to about this many triangles for a viewer; 0 returns none. The copy is geometry only."),
            new MeshFieldSpec
            {
                Name = nameof(MeshRequest.Frame),
                Access = MeshFieldAccess.Used,
                Default = nameof(MeshFrame.Native),
                Values = [nameof(MeshFrame.Native), nameof(MeshFrame.GltfYUpMetres)],
                Note = "GltfYUpMetres centres the result, scales its longest side to 1 m and turns it to glTF's front where the model's front is known.",
            },
        ];
    }

    private static MeshFieldSpec MeshField(string id) => id == MeshModelIds.Sam3dObjects
        ? Used(nameof(MeshRequest.Mesh), "false", null, null, null, "Return a triangle mesh (geometry-only GLB; the mesh decoder's vertex colours are noisy and not written) instead of splats; needs slat_decoder_mesh.safetensors beside the other checkpoints.")
        : Rejected(nameof(MeshRequest.Mesh));

    private static MeshFieldSpec ControlField(string id) => id == MeshModelIds.Omni
        ? Used(nameof(MeshRequest.Control), "required", null, null, null, "The control signal that goes with the image: a point cloud (obj, glb or ply, one token per distinct point, at most 8,192: a denser cloud is subsampled with the seed), a voxel mesh (obj or glb, sampled and quantised to 16 per axis), a bounding box (length, height, width in (0, 1]) or a skeleton (six values per bone).")
        : Rejected(nameof(MeshRequest.Control));

    private static MeshFieldSpec PreprocessField(string id) => MeshPreparationService.UsesPreparation(id)
        ? new()
        {
            Name = nameof(MeshRequest.Preprocess),
            Access = MeshFieldAccess.Used,
            Default = nameof(MeshPreprocess.None),
            Values = [nameof(MeshPreprocess.None), nameof(MeshPreprocess.Auto)],
            Note = "Auto mattes an image without a usable alpha plane (BiRefNet, loaded before the model and kept until memory is freed; counted in the peak of a model that has one) and applies this model's framing; None uses the image as given.",
        }
        : Ignored(nameof(MeshRequest.Preprocess), "The model mattes and frames its own input, or takes no image.");

    private static MeshFieldSpec Steps(string? defaultValue, double min, double? max, string note) =>
        Used(nameof(MeshRequest.Steps), defaultValue, min, max, null, note);

    private static MeshFieldSpec Grid(string? defaultValue, IReadOnlyList<string>? values, double? min, double? max, string note) =>
        new() { Name = nameof(MeshRequest.GridResolution), Access = MeshFieldAccess.Used, Default = defaultValue, Values = values, Min = min, Max = max, Note = note };

    private static MeshFieldSpec Seed(string? note = null) =>
        Used(nameof(MeshRequest.Seed), "random", 0, MeshSeed.Max, null, note ?? "A negative value draws a random seed.");

    private static MeshFieldSpec IgnoredSeed(string why) =>
        new() { Name = nameof(MeshRequest.Seed), Access = MeshFieldAccess.Ignored, Max = MeshSeed.Max, Note = $"{why} A seed above {MeshSeed.Max} is still rejected, like for every model." };

    private static MeshFieldSpec Used(string name, string? defaultValue, double? min, double? max, string? unit, string note) =>
        new()
        {
            Name = name,
            Access = MeshFieldAccess.Used,
            Default = defaultValue,
            Min = min,
            Max = max,
            Unit = unit,
            Note = note,
        };

    private static MeshFieldSpec Ignored(string name, string note) => new() { Name = name, Access = MeshFieldAccess.Ignored, Note = note };

    private static MeshFieldSpec Rejected(string name) => new() { Name = name, Access = MeshFieldAccess.Rejected, Default = DefaultOf(name) };

    private static string DefaultOf(string name) => name switch
    {
        nameof(MeshRequest.Samples) => "1",
        nameof(MeshRequest.Rigged) or nameof(MeshRequest.Texture) or nameof(MeshRequest.Mesh) => "false",
        nameof(MeshRequest.Colors) or nameof(MeshRequest.Chains) or nameof(MeshRequest.FieldOfViewRadians) or nameof(MeshRequest.Control) => "unset",
        _ => "0",
    };

    private static MeshFieldSpec[] NoHairDeck() =>
    [
        Rejected(nameof(MeshRequest.Samples)), Rejected(nameof(MeshRequest.Colors)), Rejected(nameof(MeshRequest.Chains)), Rejected(nameof(MeshRequest.Rigged)),
    ];

    private static MeshFieldSpec[] NoTexture() => [Rejected(nameof(MeshRequest.Texture)), Rejected(nameof(MeshRequest.TextureSize))];

    private static MeshFieldSpec[] TextureOptional() =>
    [
        Used(nameof(MeshRequest.Texture), "false", null, null, null, "Also generate the PBR texture; needs the texture checkpoints beside the shape ones."),
        Used(nameof(MeshRequest.TextureSize), "2048", 16, 8192, "texels", "The baked texture's edge; needs Texture. 0 takes 2048."),
    ];
}
