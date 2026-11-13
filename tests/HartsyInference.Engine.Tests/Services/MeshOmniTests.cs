using System.Text;
using System.Text.RegularExpressions;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ThreeD.Models.Hunyuan3D.Omni;
using HartsyInference.ThreeD.Pipelines;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The engine side of Hunyuan3D-Omni: its catalog entry, the request checks and the mapping of a control to the encoder's rows.</summary>
public sealed class MeshOmniTests
{
    private static ImageData Image() => new() { Rgb = new byte[4 * 4 * 3], Width = 4, Height = 4, Alpha = [.. Enumerable.Range(0, 16).Select(i => i < 8 ? (byte)0 : (byte)255)] };

    private static SceneMeshData Obj(string text) => new() { Data = Encoding.ASCII.GetBytes(text), Format = "obj" };

    private const string Triangle = "v 0 0 0\nv 2 0 0\nv 0 1 0\nf 1 2 3\n";

    private static MeshRequest Request(MeshControl? control) => new() { Image = Image(), Control = control };

    private static void Validate(MeshRequest request) => MeshRequestValidator.Validate(MeshModelIds.Omni, request, matterAvailable: false);

    [Fact]
    public void TheCatalogEntryCarriesTheOmniFilesAndDinoWithPinnedHashesAndMatchesWhatThePipelineReads()
    {
        CatalogEntry entry = ModelCatalog.Find(MeshModelIds.Omni)!;

        Assert.Equal(Modality.Mesh, entry.Modality);
        Assert.True(entry.CliDrivable);
        Assert.True(entry.LoadsFromFolder);
        Assert.Equal<string[]>([.. Hunyuan3DOmniPipeline.Files.Order(StringComparer.Ordinal)], [.. entry.Assets.Select(a => a.FileName).Order(StringComparer.Ordinal)]);
        Assert.All(entry.Assets, a => Assert.Matches(new Regex("^[0-9a-f]{64}$"), a.Sha256 ?? string.Empty));
        Assert.All(entry.Assets, a => Assert.Equal("3D/hunyuan3d-omni", a.TargetSubdir));
        Assert.Equal("model/pytorch_model.bin", ModelDownloader.PrimaryAsset(entry)!.FileName);
        Assert.Equal("facebook/dinov2-large", entry.Assets.Single(a => a.FileName == Hunyuan3DOmniPipeline.DinoFile).Repo);
        Assert.Single(entry.Assets.Where(a => a.Repo == "tencent/Hunyuan3D-Omni" && a.FileName.EndsWith(".bin", StringComparison.Ordinal)), a => a.Role == "transformer");
    }

    [Fact]
    public void OmniTakesAnImageAndAControlAndNothingElse()
    {
        MeshControl box = new() { Kind = MeshControlKind.BoundingBox, Values = [0.5f, 0.5f, 0.5f] };

        Validate(Request(box));

        Assert.Throws<NotSupportedException>(() => Validate(Request(null)));
        Assert.Throws<NotSupportedException>(() => Validate(new MeshRequest { Control = box }));
        Assert.Throws<NotSupportedException>(() => Validate(Request(box) with { SceneMesh = Obj(Triangle) }));
        Assert.Throws<NotSupportedException>(() => MeshRequestValidator.Validate("hunyuan3d", Request(box), matterAvailable: false));
        Assert.Throws<NotSupportedException>(() => Validate(Request(box) with { FieldOfViewRadians = 0.5 }));
    }

    [Theory]
    [InlineData(0, 0, true)]
    [InlineData(50, 384, true)]
    [InlineData(1000, 512, true)]
    [InlineData(1001, 0, false)]
    [InlineData(-1, 0, false)]
    [InlineData(0, 31, false)]
    [InlineData(0, 513, false)]
    public void StepsAndGridAreRangeChecked(int steps, int grid, bool accepted)
    {
        MeshRequest request = Request(new MeshControl { Kind = MeshControlKind.BoundingBox, Values = [1f, 1f, 1f] }) with { Steps = steps, GridResolution = grid };

        if (accepted)
            Validate(request);
        else
            Assert.Throws<ArgumentOutOfRangeException>(() => Validate(request));
    }

    [Fact]
    public void ControlsAreCheckedByKind()
    {
        MeshControl Point(SceneMeshData? geometry) => new() { Kind = MeshControlKind.Point, Geometry = geometry };
        MeshControl Voxel(SceneMeshData? geometry) => new() { Kind = MeshControlKind.Voxel, Geometry = geometry };
        SceneMeshData ply = new() { Data = [1], Format = "ply" };

        Validate(Request(Point(Obj(Triangle))));
        Validate(Request(Point(ply)));
        Validate(Request(Voxel(Obj(Triangle))));
        Validate(Request(new MeshControl { Kind = MeshControlKind.Pose, Values = new float[12] }));

        Assert.Throws<NotSupportedException>(() => Validate(Request(Point(null))));
        Assert.Throws<NotSupportedException>(() => Validate(Request(Voxel(ply))));
        Assert.Throws<NotSupportedException>(() => Validate(Request(Point(Obj(Triangle)) with { Values = [1f] })));
        Assert.Throws<NotSupportedException>(() => Validate(Request(new MeshControl { Kind = MeshControlKind.BoundingBox, Geometry = Obj(Triangle), Values = [1f, 1f, 1f] })));
        Assert.Throws<NotSupportedException>(() => Validate(Request(new MeshControl { Kind = MeshControlKind.Pose })));
        Assert.Throws<ArgumentException>(() => Validate(Request(new MeshControl { Kind = MeshControlKind.BoundingBox, Values = [0.5f, 0.5f] })));
        Assert.Throws<ArgumentException>(() => Validate(Request(new MeshControl { Kind = MeshControlKind.BoundingBox, Values = [0.5f, 0f, 1f] })));
        Assert.Throws<ArgumentException>(() => Validate(Request(new MeshControl { Kind = MeshControlKind.BoundingBox, Values = [0.5f, 1.5f, 1f] })));
        Assert.Throws<ArgumentException>(() => Validate(Request(new MeshControl { Kind = MeshControlKind.Pose, Values = new float[7] })));
        Assert.Throws<ArgumentException>(() => Validate(Request(new MeshControl { Kind = MeshControlKind.Pose, Values = [.. new float[5], float.NaN] })));
        Assert.Throws<ArgumentOutOfRangeException>(() => Validate(Request(new MeshControl { Kind = (MeshControlKind)9, Values = new float[6] })));
    }

    [Fact]
    public void AControlGivenToAnotherModelNamesTheField()
    {
        NotSupportedException error = Assert.Throws<NotSupportedException>(() =>
            MeshRequestValidator.Validate("triposr", new MeshRequest { Image = Image(), Control = new MeshControl { Kind = MeshControlKind.BoundingBox, Values = [1f, 1f, 1f] } }, matterAvailable: false));

        Assert.Contains("Control", error.Message);
        Assert.Equal(nameof(MeshRequest.Control), MeshFieldTag.FieldOf(error));
    }

    [Fact]
    public void ToControl_BuildsTheRowsOfEachKind()
    {
        OmniControl point = OmniControlMapper.ToControl(new MeshControl { Kind = MeshControlKind.Point, Geometry = Obj(Triangle) }, 1);
        OmniControl voxel = OmniControlMapper.ToControl(new MeshControl { Kind = MeshControlKind.Voxel, Geometry = Obj(Triangle) }, 1);
        OmniControl box = OmniControlMapper.ToControl(new MeshControl { Kind = MeshControlKind.BoundingBox, Values = [0.5f, 0.25f, 1f] }, 1);
        OmniControl pose = OmniControlMapper.ToControl(new MeshControl { Kind = MeshControlKind.Pose, Values = [1f, 2f, 3f, 4f, 5f, 6f] }, 1);

        Assert.Equal((OmniControlKind.Point, 3 * 6), (point.Kind, point.Rows.Length));
        Assert.Equal(OmniControlKind.Voxel, voxel.Kind);
        Assert.True(voxel.Rows.Length > 6 && voxel.Rows.Length % 6 == 0);
        Assert.Equal([0.5f, 0.25f, 1f, 0.5f, 0.25f, 1f], box.Rows);
        Assert.Equal([1f, 2f, 3f, 4f, 5f, 6f], pose.Rows);
        // The same seed gives the same voxels.
        Assert.Equal(voxel.Rows, OmniControlMapper.ToControl(new MeshControl { Kind = MeshControlKind.Voxel, Geometry = Obj(Triangle) }, 1).Rows);
    }

    [Fact]
    public void ToControl_ReadsAPointCloudPlyAndRefusesAVoxelMeshWithoutFaces()
    {
        string ply = "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n2 0 0\n0 1 0\n";

        OmniControl point = OmniControlMapper.ToControl(new MeshControl { Kind = MeshControlKind.Point, Geometry = new SceneMeshData { Data = Encoding.ASCII.GetBytes(ply), Format = "ply" } }, 1);

        Assert.Equal(3 * 6, point.Rows.Length);
        // An obj with vertices only is a point cloud too.
        Assert.Equal(3 * 6, OmniControlMapper.ToControl(new MeshControl { Kind = MeshControlKind.Point, Geometry = Obj("v 0 0 0\nv 1 0 0\nv 0 1 0\n") }, 1).Rows.Length);
        Assert.Throws<NotSupportedException>(() => OmniControlMapper.ToControl(new MeshControl { Kind = MeshControlKind.Voxel, Geometry = Obj("v 0 0 0\nv 1 0 0\nv 0 1 0\n") }, 1));
    }

    [Fact]
    public void AMalformedPointFileIsReportedAsAnUnusableControlNamingTheField()
    {
        MeshControl Point(string format, byte[] data) => new() { Kind = MeshControlKind.Point, Geometry = new SceneMeshData { Data = data, Format = format } };

        foreach (MeshControl control in new[]
        {
            Point("ply", Encoding.ASCII.GetBytes("not a ply")),
            Point("obj", Encoding.ASCII.GetBytes("v 1 2 x\n")),
            Point("obj", Encoding.ASCII.GetBytes("v 1 1 1\n")),
            Point("glb", [1, 2, 3, 4]),
        })
        {
            NotSupportedException error = Assert.Throws<NotSupportedException>(() => OmniControlMapper.ToControl(control, 1));
            Assert.Equal(nameof(MeshRequest.Control), MeshFieldTag.FieldOf(error));
        }
    }

    [Fact]
    public void APointControlRoundTripsThroughObjAndPlyWithTheSameRows()
    {
        string obj = "v 0 0 0\nv 2 0 0\nv 0 1 0\nf 1 2 3\n";
        string ply = "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n2 0 0\n0 1 0\n";

        float[] fromObj = OmniControlMapper.ToControl(new MeshControl { Kind = MeshControlKind.Point, Geometry = Obj(obj) }, 1).Rows;
        float[] fromPly = OmniControlMapper.ToControl(new MeshControl { Kind = MeshControlKind.Point, Geometry = new SceneMeshData { Data = Encoding.ASCII.GetBytes(ply), Format = "ply" } }, 1).Rows;

        Assert.Equal(fromObj, fromPly);
    }

    [Fact]
    public void TheProfileDescribesTheControlAsRequiredAndOnlyOmniAcceptsIt()
    {
        Assert.Equal(MeshFieldAccess.Used, MeshModelProfiles.Find(MeshModelIds.Omni)!.Fields.Single(f => f.Name == nameof(MeshRequest.Control)).Access);
        foreach (string id in MeshModelIds.Described.Where(id => id != MeshModelIds.Omni))
            Assert.Equal(MeshFieldAccess.Rejected, MeshModelProfiles.Find(id)!.Fields.Single(f => f.Name == nameof(MeshRequest.Control)).Access);
        Assert.Equal(MeshInputKind.ImageAndControl, MeshModelProfiles.Find(MeshModelIds.Omni)!.Input);
        Assert.Contains(MeshModelIds.Omni, MeshModelIds.All);
    }
}
