using System.Text.RegularExpressions;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Models.Sam3dObjects;
using HartsyInference.ThreeD.Pipelines;

namespace HartsyInference.Engine.Tests.Services;

public sealed class Sam3dObjectsWiringTests
{
    private static readonly Regex Sha256 = new("^[0-9a-f]{64}$", RegexOptions.Compiled);

    private static ImageData Image(bool alpha = true, byte maskValue = 255)
    {
        byte[] mask = new byte[4 * 3];
        Array.Fill(mask, maskValue);
        return new ImageData { Rgb = new byte[4 * 3 * 3], Width = 4, Height = 3, Alpha = alpha ? mask : null };
    }

    [Fact]
    public void Entry_IsAHashedMeshEntryWithTheSafetensorsSetMogeAndTheLicence()
    {
        CatalogEntry? entry = ModelCatalog.Find("sam3d-objects");

        Assert.NotNull(entry);
        Assert.Equal(Modality.Mesh, entry.Modality);
        Assert.True(entry.CliDrivable);
        Assert.All(entry.Assets, asset => Assert.Matches(Sha256, asset.Sha256 ?? string.Empty));
        Assert.All(entry.Assets, asset => Assert.Equal("3D/sam3d-objects", asset.TargetSubdir));
        string[] files = [.. entry.Assets.Select(a => a.RepoPath)];
        Assert.Contains("ss_generator.safetensors", files);
        Assert.Contains("ss_decoder.safetensors", files);
        Assert.Contains("slat_generator.safetensors", files);
        Assert.Contains("slat_decoder_gs.safetensors", files);
        Assert.Contains("moge_vitl.safetensors", files);
        // The SAM License binds on use and on redistribution: it is fetched beside the weights. The mesh decoder is part of the entry, for the mesh output.
        Assert.Contains("SAM3D_LICENSE", files);
        Assert.Contains("slat_decoder_mesh.safetensors", files);
    }

    [Fact]
    public void Model_IsWiredToTheMeshServiceAsAnImageModel()
    {
        Assert.Contains("sam3d-objects", MeshModelIds.ImageToMesh);
        Assert.Contains("sam3d-objects", MeshModelIds.All);
        Assert.Equal("sam3d-objects", ModelResolver.Resolve("sam3d-objects", null, Modality.Mesh).Catalog?.Id);
    }

    [Fact]
    public void ToRequest_TakesTheAlphaPlaneAsTheMaskAndRunsBothFlowsAtTheDefaultSteps()
    {
        MeshRequest request = new() { Image = Image(), Seed = 7 };

        Sam3dObjectsRequest job = Sam3dObjectsRequestMapper.ToRequest(request);

        Assert.Same(request.Image!.Alpha, job.Mask);
        Assert.Same(request.Image.Rgb, job.Rgb);
        Assert.Equal((4, 3), (job.Width, job.Height));
        Assert.Null(job.Pointmap);
        Assert.Equal(7, job.Seed);
        Assert.Equal((25, 25), (job.StructureSteps, job.SlatSteps));
    }

    [Fact]
    public void ToRequest_UsesTheStepCountForBothFlowsAndARandomSeedForANegativeOne()
    {
        Sam3dObjectsRequest job = Sam3dObjectsRequestMapper.ToRequest(new MeshRequest { Image = Image(), Steps = 10 });

        Assert.Equal((10, 10), (job.StructureSteps, job.SlatSteps));
        Assert.Null(job.Seed);
    }

    [Fact]
    public void AlphaFromMask_MapsEveryNonZeroRedValueToFullCoverage()
    {
        // A 0/1 label mask is the object, not a 1/255 sliver of it; the green and blue channels are not read.
        byte[] rgb = [0, 9, 9, 1, 0, 0, 255, 0, 0, 128, 7, 7];

        byte[] alpha = Sam3dObjectsRequestMapper.AlphaFromMask(rgb, 4, 1);

        Assert.Equal([(byte)0, (byte)255, (byte)255, (byte)255], alpha);
        Assert.Throws<ArgumentException>(() => Sam3dObjectsRequestMapper.AlphaFromMask(rgb, 3, 1));
    }

    [Fact]
    public void Validate_RejectsAMissingImageMaskOrAnEmptyMask()
    {
        Assert.Throws<NotSupportedException>(() => Sam3dObjectsRequestMapper.Validate(new MeshRequest()));
        NotSupportedException noMask = Assert.Throws<NotSupportedException>(() => Sam3dObjectsRequestMapper.Validate(new MeshRequest { Image = Image(alpha: false) }));
        Assert.Contains("mask", noMask.Message);
        Assert.Throws<NotSupportedException>(() => Sam3dObjectsRequestMapper.Validate(new MeshRequest { Image = Image(maskValue: 0) }));
    }

    [Fact]
    public void Validate_RejectsTheFieldsSam3dDoesNotRead()
    {
        Assert.Throws<NotSupportedException>(() => Sam3dObjectsRequestMapper.Validate(new MeshRequest { Image = Image(), GridResolution = 512 }));
        Assert.Throws<NotSupportedException>(() => Sam3dObjectsRequestMapper.Validate(new MeshRequest { Image = Image(), Texture = true }));
        Assert.Throws<NotSupportedException>(() => Sam3dObjectsRequestMapper.Validate(new MeshRequest { Image = Image(), TextureSize = 1024 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => Sam3dObjectsRequestMapper.Validate(new MeshRequest { Image = Image(), Steps = -1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => Sam3dObjectsRequestMapper.Validate(new MeshRequest { Image = Image(), Seed = (long)int.MaxValue + 1 }));
        Sam3dObjectsRequestMapper.Validate(new MeshRequest { Image = Image() });
    }

    [Fact]
    public async Task GenerateAsync_RejectsARequestWithoutAMaskBeforeAnyCheckpointLoads()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        ModelSpec spec = ModelResolver.Resolve("sam3d-objects", "does-not-exist", Modality.Mesh);

        NotSupportedException error = await Assert.ThrowsAsync<NotSupportedException>(
            () => engine.Mesh.GenerateAsync(spec, new MeshRequest { Image = Image(alpha: false) }));

        Assert.Contains("mask", error.Message);
    }

    [Fact]
    public async Task GenerateAsync_StillRunsAfterTheServiceWasDisposedToFreeMemory()
    {
        string folder = Directory.CreateTempSubdirectory("sam3d-wiring").FullName;
        try
        {
            using InferenceEngine engine = new InferenceEngine("cpu");
            ModelSpec spec = ModelResolver.Resolve("sam3d-objects", folder, Modality.Mesh);
            // Empty checkpoint files get a request past the file checks and through the queue, where it fails reading them, not for being unloaded.
            foreach (string name in new[] { "ss_generator", "ss_decoder", "slat_generator", "slat_decoder_gs", "moge_vitl" })
                File.WriteAllBytes(Path.Combine(folder, name + ".safetensors"), []);
            // The engine disposes the service to free memory and keeps the instance: later requests must not be rejected as unloaded.
            ((IDisposable)engine.Mesh).Dispose();

            Exception error = await Record.ExceptionAsync(() => engine.Mesh.GenerateAsync(spec, new MeshRequest { Image = Image() }));

            Assert.NotNull(error);
            Assert.IsNotType<ModelUnloadedException>(error);
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    [Fact]
    public async Task GenerateAsync_NamesTheMissingCheckpointsOfAnIncompleteFolder()
    {
        string folder = Directory.CreateTempSubdirectory("sam3d-wiring").FullName;
        try
        {
            using InferenceEngine engine = new InferenceEngine("cpu");
            ModelSpec spec = ModelResolver.Resolve("sam3d-objects", folder, Modality.Mesh);

            FileNotFoundException error = await Assert.ThrowsAsync<FileNotFoundException>(
                () => engine.Mesh.GenerateAsync(spec, new MeshRequest { Image = Image() }));

            Assert.Contains("ss_generator", error.Message);
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    [Fact]
    public void MeshOption_IsPassedToThePipelineAndRejectedForOtherModels()
    {
        MeshRequest request = new() { Image = Image(), Mesh = true };

        Assert.True(Sam3dObjectsRequestMapper.ToRequest(request).Mesh);
        Assert.False(Sam3dObjectsRequestMapper.ToRequest(request with { Mesh = false }).Mesh);
        NotSupportedException error = Assert.Throws<NotSupportedException>(() => MeshRequestValidator.Validate("triposr", new MeshRequest { Image = Image(), Mesh = true }, matterAvailable: false));
        Assert.Equal(nameof(MeshRequest.Mesh), MeshFieldTag.FieldOf(error));
    }

    [Fact]
    public void ToMesh_TurnsTheExtractionFrameToGltfAndTakesTheFirstThreeColourChannels()
    {
        SamMeshResult result = new([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f], [0, 1, 2], [0.1f, 0.2f, 0.3f, 0.9f, 0.9f, 0.9f, 0.4f, 0.5f, 0.6f, 0.9f, 0.9f, 0.9f, 0.7f, 0.8f, 0.9f, 0.9f, 0.9f, 0.9f]);

        Mesh mesh = Sam3dObjectsMesh.ToMesh(result);

        // (x, y, z) becomes (x, z, -y), found on the real bust; the winding is kept.
        Assert.Equal([1f, 3f, -2f, 4f, 6f, -5f, 7f, 9f, -8f], mesh.Vertices);
        Assert.Equal([0, 1, 2], mesh.Indices);
        Assert.Equal([0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f], mesh.VertexColors!);
    }

    [Fact]
    public void ToMesh_RefusesColoursOfTheWrongLength()
    {
        SamMeshResult result = new([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f], [0, 1, 2], [0.1f, 0.2f, 0.3f]);

        Assert.Throws<ArgumentException>(() => Sam3dObjectsMesh.ToMesh(result));
    }

    [Fact]
    public void Finish_DropsAFloatingPieceOfAbout3In10000OfTheAreaThatTheSharedDefaultKeeps()
    {
        // A tetrahedron of unit edges (area about 1.73) and a far triangle of area 0.0005, so 3e-4 of the total: above the default 1e-5, below the 1e-3 used here.
        Mesh mesh = new()
        {
            Vertices = [0f, 0f, 0f, 1f, 0f, 0f, 0.5f, 0.866f, 0f, 0.5f, 0.289f, 0.816f, 5f, 5f, 5f, 5.0316f, 5f, 5f, 5f, 5.0316f, 5f],
            Indices = [0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3, 4, 5, 6],
        };
        Assert.Equal(2, MeshStatistics.Measure(ShapeMeshFinisher.Finish(mesh)).Components);

        MeshStats finished = MeshStatistics.Measure(Sam3dObjectsMesh.Finish(mesh));

        Assert.Equal(1, finished.Components);
        Assert.Equal(4, finished.TriangleCount);
    }
}
