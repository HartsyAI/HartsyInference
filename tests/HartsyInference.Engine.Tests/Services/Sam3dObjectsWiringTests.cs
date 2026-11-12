using System.Text.RegularExpressions;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
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
        // The SAM License binds on use and on redistribution: it is fetched beside the weights. The unported mesh decoder is not part of the entry.
        Assert.Contains("SAM3D_LICENSE", files);
        Assert.DoesNotContain(files, f => f.Contains("mesh", StringComparison.OrdinalIgnoreCase));
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
}
