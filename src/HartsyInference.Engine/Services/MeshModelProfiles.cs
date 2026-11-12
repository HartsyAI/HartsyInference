using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>The profiles of every described 3D model. The rules they state are the ones <see cref="MeshRequestValidator"/> enforces; a test walks the legal values and the first illegal ones of each field through the validator.</summary>
internal static class MeshModelProfiles
{
    private const string Hunyuan2Terms = "Tencent Hunyuan 3D 2.0 Community License";
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
                    Rejected(nameof(MeshRequest.FieldOfViewRadians)), Ignored(nameof(MeshRequest.Seed), "There is no sampler to seed."),
                    .. NoHairDeck(), .. NoTexture()],
                new MeshLicence { Summary = "MIT (stabilityai/TripoSR)" },
                new MeshCost { Basis = "Verified on CPU and an RTX 3060; no time or memory figure is recorded." },
                note: "Expects the foreground cut out and composited on mid-gray."),
            .. HunyuanShapeProfiles(),
            Profile("trellis", MeshInputKind.Image, MeshOutputKinds.Splats,
                [Steps("25", 1, null, "Sampler steps of both flows."), Ignored(nameof(MeshRequest.GridResolution), "The output is a splat cloud."),
                    Rejected(nameof(MeshRequest.FieldOfViewRadians)), WrappingSeed(), .. NoHairDeck(), .. NoTexture()],
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
            Profile(MeshModelIds.Sam3dObjects, MeshInputKind.ImageWithMask, MeshOutputKinds.Splats,
                [Steps("25", 0, null, "Steps of the structure flow and of the SLAT flow."), Rejected(nameof(MeshRequest.GridResolution)),
                    Rejected(nameof(MeshRequest.FieldOfViewRadians)), Seed(), .. NoHairDeck(), .. NoTexture()],
                new MeshLicence
                {
                    Summary = "SAM License (binds on use and redistribution)",
                    Restrictions = ["Use ends for trade-controlled parties and for military, nuclear, espionage and weapons end uses; read the licence text fetched beside the weights"],
                },
                new MeshCost { TypicalSeconds = 450, Basis = "RTX 3060, F32 flows, one example image; the SLAT flow fills the card, so no peak is stated." },
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
                    Rejected(nameof(MeshRequest.FieldOfViewRadians)), Ignored(nameof(MeshRequest.Seed), "The decomposer is deterministic."), .. NoHairDeck(), .. NoTexture()],
                new MeshLicence { Summary = "Not recorded in the engine's documents" },
                new MeshCost { Basis = "No time or memory figure is recorded." },
                note: "Takes a scene mesh and, optionally, a mask or point prompt in ScenePrompt."),
            Profile("unirig", MeshInputKind.SceneMesh, MeshOutputKinds.RiggedMesh,
                [Rejected(nameof(MeshRequest.Steps)), Rejected(nameof(MeshRequest.GridResolution)), Rejected(nameof(MeshRequest.FieldOfViewRadians)),
                    WrappingSeed(), .. NoHairDeck(), .. NoTexture()],
                new MeshLicence { Summary = "MIT (code and weights); training data CC-BY-4.0" },
                new MeshCost { Basis = "CPU F32 skinning of one mesh took 400 s; no GPU figure is recorded." },
                note: "Skeleton and skin weights for a triangle mesh; the result is a rigged glb."),
            Profile("hunyuan3d-part", MeshInputKind.SceneMesh, MeshOutputKinds.PartMesh,
                [Rejected(nameof(MeshRequest.Steps)), Rejected(nameof(MeshRequest.GridResolution)), Rejected(nameof(MeshRequest.FieldOfViewRadians)),
                    Used(nameof(MeshRequest.Seed), "random", 0, int.MaxValue, null, "Seeds the surface sampling."), .. NoHairDeck(), .. NoTexture()],
                new MeshLicence
                {
                    Summary = "Tencent Hunyuan 3D-Part Community License; the Sonata weights' own terms are unverified",
                    Restrictions = HunyuanRestrictions,
                },
                new MeshCost { PeakVramGiB = 4.7, TypicalSeconds = 53, Basis = "RTX 3060 through the CLI on a 1.0 M-face bust, peak 4.7." },
                peak: _ => 4.7,
                note: "Segments a mesh into parts; the labels and boxes come back in MeshResult.Parts."),
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
                    Rejected(nameof(MeshRequest.FieldOfViewRadians)), WrappingSeed(), .. NoHairDeck(), .. NoTexture()],
                new MeshLicence { Summary = Hunyuan2Terms, Restrictions = HunyuanRestrictions },
                new MeshCost { Basis = "Verified stage by stage against the Python reference; the recorded times are historical and differ by build." },
                note: "Expects the foreground recentred on white at 512 px; the engine does not prepare the image yet.");
        }
    }

    private static MeshModelProfile Paint(string id, string terms, string note, MeshCost cost, Func<MeshRequest, double?>? peak) =>
        Profile(id, MeshInputKind.SceneMeshAndImage, MeshOutputKinds.TexturedMesh,
            [Steps(null, 0, null, "Sampler steps; 0 takes the checkpoint's default."),
                Used(nameof(MeshRequest.GridResolution), "2048", 0, null, "texels", "The baked texture's edge, not a grid; 0 takes 2048."),
                Rejected(nameof(MeshRequest.FieldOfViewRadians)), Used(nameof(MeshRequest.Seed), "random", 0, int.MaxValue, null, "Seeds the multiview noise."),
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
        new() { Id = id, Input = input, Outputs = outputs, Fields = [.. fields, PreprocessField(id)], Licence = licence, Cost = cost, PeakGiB = peak, Note = note };

    private static MeshFieldSpec PreprocessField(string id) => MeshPreparationService.UsesPreparation(id)
        ? new()
        {
            Name = nameof(MeshRequest.Preprocess),
            Access = MeshFieldAccess.Used,
            Default = nameof(MeshPreprocess.None),
            Values = [nameof(MeshPreprocess.None), nameof(MeshPreprocess.Auto)],
            Note = "Auto mattes an image without a usable alpha plane (BiRefNet, loaded before the model and kept until memory is freed, not counted in the peak) and applies this model's framing; None uses the image as given.",
        }
        : Ignored(nameof(MeshRequest.Preprocess), "The model mattes and frames its own input, or takes no image.");

    private static MeshFieldSpec Steps(string? defaultValue, double min, double? max, string note) =>
        Used(nameof(MeshRequest.Steps), defaultValue, min, max, null, note);

    private static MeshFieldSpec Grid(string? defaultValue, IReadOnlyList<string>? values, double? min, double? max, string note) =>
        new() { Name = nameof(MeshRequest.GridResolution), Access = MeshFieldAccess.Used, Default = defaultValue, Values = values, Min = min, Max = max, Note = note };

    private static MeshFieldSpec WrappingSeed() =>
        Used(nameof(MeshRequest.Seed), "random", 0, null, null, "A negative value draws a random seed. Only the low 32 bits are used: a larger value wraps instead of failing.");

    private static MeshFieldSpec Seed() =>
        Used(nameof(MeshRequest.Seed), "random", 0, int.MaxValue, null, "A negative value draws a random seed.");

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
        nameof(MeshRequest.Rigged) or nameof(MeshRequest.Texture) => "false",
        nameof(MeshRequest.Colors) or nameof(MeshRequest.Chains) or nameof(MeshRequest.FieldOfViewRadians) => "unset",
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
