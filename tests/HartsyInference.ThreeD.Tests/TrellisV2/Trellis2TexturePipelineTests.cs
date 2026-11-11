using System.Numerics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Models.TrellisV2;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;
using HartsyInference.ThreeD.Pipelines.Stages;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

/// <summary>The TRELLIS.2 texture path over stub stages: stage order and seeds per pipeline type, which conditioning and flow the texture
/// stage gets, request checks before any stage runs, progress, cancellation with tensor cleanup, the volume handed to the baker, and the
/// GLB frame. The pipeline.json texture entries and the texture file lookup are checked alongside.</summary>
public sealed class Trellis2TexturePipelineTests
{
    private static readonly string ZeroStats = "[" + string.Join(",", Enumerable.Repeat("0", 32)) + "]";
    private static readonly string OneStats = "[" + string.Join(",", Enumerable.Repeat("1", 32)) + "]";

    private static string Json(string textureStd) => $$$"""
        {"name": "Trellis2ImageTo3DPipeline", "args": {
          "sparse_structure_sampler": {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5},
            "params": {"steps": 3, "guidance_strength": 7.5, "guidance_rescale": 0.7, "guidance_interval": [0.6, 1.0], "rescale_t": 5.0}},
          "shape_slat_sampler": {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5},
            "params": {"steps": 2, "guidance_strength": 7.5, "guidance_rescale": 0.5, "guidance_interval": [0.6, 1.0], "rescale_t": 3.0}},
          "shape_slat_normalization": {"mean": {{{ZeroStats}}}, "std": {{{OneStats}}}},
          "tex_slat_sampler": {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5},
            "params": {"steps": 2, "guidance_strength": 1.0, "guidance_rescale": 0.0, "guidance_interval": [0.6, 0.9], "rescale_t": 3.0}},
          "tex_slat_normalization": {"mean": {{{ZeroStats}}}, "std": {{{textureStd}}} } } }
        """;

    private static Trellis2PipelineConfig Config => Trellis2PipelineConfig.FromJson(Json(OneStats));

    private static Trellis2InputImage Image() => new()
    {
        Rgb = new byte[8 * 8 * 3],
        Alpha = Enumerable.Range(0, 64).Select(i => i is >= 18 and <= 45 ? (byte)255 : (byte)0).ToArray(),
        Width = 8,
        Height = 8,
    };

    private static Trellis2Options Options(int grid = 512, int seed = 100) => new() { Grid = grid, Seed = seed, Texture = true, TextureSize = 16 };

    [Fact]
    public void GenerateTextured_Grid512RunsTheTexture512FlowAfterTheShapeDecode()
    {
        using Rig rig = new();

        PbrPostProcessResult result = rig.Pipeline.GenerateTextured(Image(), Options());

        Assert.Equal(
            [
                "cond@512", "structure@32 steps=3 seed=100", "shape Res512 grid=32 seed=101 steps=2", "decode+subs",
                "texture Res512 voxels=2 seed=103 steps=2", "texture-decode levels=1",
            ],
            rig.Calls);
        Assert.Same(rig.CondByResolution[512], rig.TextureCond);
        Assert.NotNull(result.Mesh.Uvs);
        Assert.Equal((16, 16), (result.Material.BaseColor.Width, result.Material.BaseColor.Height));
    }

    [Fact]
    public void GenerateTextured_CascadeUsesThe1024FlowAndTheFineConditioning()
    {
        using Rig rig = new();

        rig.Pipeline.GenerateTextured(Image(), Options(grid: 1024));

        Assert.Equal(
            [
                "cond@512", "structure@32 steps=3 seed=100", "shape Res512 grid=32 seed=101 steps=2", "upsample x4",
                "cond@1024", "shape Res1024 grid=64 seed=102 steps=2", "decode+subs", "texture Res1024 voxels=2 seed=103 steps=2",
                "texture-decode levels=1",
            ],
            rig.Calls);
        Assert.Same(rig.CondByResolution[1024], rig.TextureCond);
    }

    [Fact]
    public void GenerateTextured_StepsOverrideTheTextureSamplerToo()
    {
        using Rig rig = new();

        rig.Pipeline.GenerateTextured(Image(), Options() with { Steps = 5 });

        Assert.Contains("texture Res512 voxels=2 seed=103 steps=5", rig.Calls);
    }

    [Fact]
    public void Generate_StaysShapeOnlyEvenWhenTextureIsRequested()
    {
        using Rig rig = new();

        Mesh mesh = rig.Pipeline.Generate(Image(), Options());

        Assert.Equal(["cond@512", "structure@32 steps=3 seed=100", "shape Res512 grid=32 seed=101 steps=2", "decode"], rig.Calls);
        Assert.Equal(2, mesh.TriangleCount);
    }

    [Fact]
    public void GenerateTextured_WithoutTextureStagesFailsBeforeAnyStageRuns()
    {
        using Rig rig = new(texture: false);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => rig.Pipeline.GenerateTextured(Image(), Options()));

        Assert.Contains("texture", error.Message, StringComparison.OrdinalIgnoreCase);
        Assert.False(rig.Pipeline.HasTexture);
        Assert.Empty(rig.Calls);
    }

    [Fact]
    public void GenerateTextured_WithoutTheNeededFlowFailsBeforeAnyStageRuns()
    {
        using Rig rig = new(supports1024: false);

        Assert.Throws<HartsyInferenceException>(() => rig.Pipeline.GenerateTextured(Image(), Options(grid: 1536)));

        Assert.Empty(rig.Calls);
        // The 512 type needs only the 512 flow.
        rig.Pipeline.GenerateTextured(Image(), Options());
        Assert.Contains("texture-decode levels=1", rig.Calls);
    }

    [Theory]
    [InlineData(8)]
    [InlineData(-16)]
    [InlineData(8193)]
    public void GenerateTextured_RejectsABadTextureSizeBeforeAnyStageRuns(int size)
    {
        using Rig rig = new();

        Assert.Throws<ArgumentException>(() => rig.Pipeline.GenerateTextured(Image(), Options() with { TextureSize = size }));
        Assert.Throws<ArgumentException>(() => rig.Pipeline.GenerateTextured(Image(), Options() with { DecimationTarget = -1 }));

        Assert.Empty(rig.Calls);
    }

    [Fact]
    public void GenerateTextured_ReportsMonotonicProgressEndingAtTheTotal()
    {
        using Rig rig = new();
        List<GenerationProgress> reports = [];

        rig.Pipeline.GenerateTextured(Image(), Options(), reports.Add);

        // 512 type: 1 + 3 + 2 + 1 (shape decode) + 2 (texture steps) + 1 (texture decode) + 1 (bake) = 11.
        Assert.Equal(Enumerable.Range(1, 11), reports.Select(p => p.Step));
        Assert.All(reports, p => Assert.Equal(11, p.TotalSteps));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(10)]
    public void GenerateTextured_PropagatesACancelAndReleasesEveryTensor(int cancelAtStep)
    {
        using Rig rig = new();

        Assert.Throws<OperationCanceledException>(() => rig.Pipeline.GenerateTextured(Image(), Options(), p =>
        {
            if (p.Step == cancelAtStep)
                throw new OperationCanceledException();
        }));

        Assert.NotEmpty(rig.Tensors);
        foreach (Tensor t in rig.Tensors)
            Trellis2TestData.AssertDisposed(t);
    }

    [Fact]
    public void GenerateTextured_ReleasesEveryTensorAfterASuccessfulRun()
    {
        using Rig rig = new();

        rig.Pipeline.GenerateTextured(Image(), Options());

        Assert.NotEmpty(rig.Tensors);
        foreach (Tensor t in rig.Tensors)
            Trellis2TestData.AssertDisposed(t);
    }

    [Fact]
    public void GenerateTextured_BakesTheDecodedAttributesIntoTheMaterial()
    {
        using Rig rig = new();

        PbrPostProcessResult result = rig.Pipeline.GenerateTextured(Image(), Options());

        Assert.True(result.CoveredTexels > 0);
        Assert.NotNull(result.Material.Alpha);
        RasterizedUvMap raster = Rasterizer.Rasterize(result.Mesh.Uvs!, result.Mesh.Indices, 16, 16);
        int covered = 0;
        for (int texel = 0; texel < 16 * 16; texel++)
        {
            if (raster.TriangleIds[texel] < 0)
                continue;
            covered++;
            byte[] color = result.Material.BaseColor.Pixels;
            Assert.InRange(color[texel * 3], 202, 206);
            Assert.InRange(color[texel * 3 + 1], 100, 104);
            Assert.InRange(color[texel * 3 + 2], 49, 53);
            Assert.InRange(result.Material.Metallic.Pixels[texel], 62, 66);
            Assert.InRange(result.Material.Roughness.Pixels[texel], 189, 193);
        }

        Assert.True(covered > 0);
    }

    [Fact]
    public void ToGlb_IsAGlbInGltfsYUpFrame()
    {
        using Rig rig = new();
        PbrPostProcessResult result = rig.Pipeline.GenerateTextured(Image(), Options());

        byte[] glb = Trellis2ImageTo3DPipeline.ToGlb(result);

        Assert.Equal("glTF", System.Text.Encoding.ASCII.GetString(glb, 0, 4));
        Mesh read = GlbReader.Parse(glb);
        Assert.Equal(result.Mesh.TriangleCount, read.TriangleCount);
        // (x, y, z) -> (x, z, -y): the Z-up mesh's first vertex lands with its old z in the y slot and minus its old y in z.
        Assert.Equal(result.Mesh.Vertices[0], read.Vertices[0], 5);
    }

    [Fact]
    public void BuildVolume_PlacesTheVoxelsOnTheMeshGrid()
    {
        Tensor feats = new(new TensorShape(1, 2, 6), DType.F32);
        float[] values = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f];
        values.CopyTo(feats.AsSpan<float>());
        SparseTensor voxels = new(feats, [0, 3, 4, 5, 0, 6, 7, 8], 64);

        SparseAttributeVolume volume = Trellis2ImageTo3DPipeline.BuildVolume(voxels, 64);

        Assert.Equal(6, volume.Channels);
        Assert.Equal((64, 64, 64), volume.GridSize);
        Assert.Equal(new Vector3(-0.5f), volume.Origin);
        Assert.Equal(1f / 64f, volume.VoxelSize);
        Assert.True(volume.TryFindIndex(6, 7, 8, out int index));
        Assert.Equal(0.9f, volume.Attributes[index * 6]);
        Assert.False(volume.TryFindIndex(5, 4, 3, out _));
        Assert.Throws<HartsyInferenceException>(() => Trellis2ImageTo3DPipeline.BuildVolume(voxels, 128));
        feats.Dispose();
    }

    [Fact]
    public void BuildVolume_SampledAtAVoxelCenterReturnsThatVoxelsAttributes()
    {
        Tensor feats = new(new TensorShape(1, 1, 6), DType.F32);
        new float[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f }.CopyTo(feats.AsSpan<float>());
        SparseAttributeVolume volume = Trellis2ImageTo3DPipeline.BuildVolume(new SparseTensor(feats, [0, 10, 11, 12], 32), 32);
        float voxel = 1f / 32f;
        Span<float> sample = stackalloc float[6];

        bool hit = SparseAttributeSampler.SampleWorld(
            volume, -0.5 + 10.5 * voxel, -0.5 + 11.5 * voxel, -0.5 + 12.5 * voxel, sample);

        Assert.True(hit);
        Assert.Equal(0.4f, sample[3], 6);
        feats.Dispose();
    }

    [Fact]
    public void Constructor_RequiresTheTextureStagesTogetherAndASubdivisionRecordingDecoder()
    {
        using CpuBackend backend = new();
        Rig rig = new();

        Assert.Throws<ArgumentException>(() => rig.Build(backend, textureFlow: new StubTextureFlow(rig, true), textureDecoder: null));
        Assert.Throws<ArgumentException>(
            () => rig.Build(backend, new PlainDecoder(), new StubTextureFlow(rig, true), new StubTextureDecoder(rig)));
        rig.Dispose();
    }

    [Fact]
    public void PipelineConfig_ParsesTheShippedTextureSamplerAndNormalization()
    {
        Trellis2PipelineConfig config = Config;

        Assert.True(config.HasTexture);
        Assert.Equal(2, config.TextureSampler!.Steps);
        Assert.Equal(1f, config.TextureSampler.GuidanceStrength);
        Assert.Equal(0f, config.TextureSampler.GuidanceRescale);
        Assert.Equal((0.6, 0.9), (config.TextureSampler.IntervalLow, config.TextureSampler.IntervalHigh));
        Assert.Equal(3.0, config.TextureSampler.RescaleT);
        Assert.Equal(32, config.TextureLatentMean!.Length);
        Assert.Equal(1f, config.TextureLatentStd![31]);
    }

    [Fact]
    public void PipelineConfig_WithoutTextureEntriesIsShapeOnly()
    {
        string json = Json(OneStats).Replace("\"tex_slat_sampler\"", "\"unused_sampler\"", StringComparison.Ordinal)
            .Replace("\"tex_slat_normalization\"", "\"unused_normalization\"", StringComparison.Ordinal);

        Trellis2PipelineConfig config = Trellis2PipelineConfig.FromJson(json);

        Assert.False(config.HasTexture);
        Assert.Null(config.TextureSampler);
        Assert.Null(config.TextureLatentMean);
    }

    [Fact]
    public void PipelineConfig_RejectsATextureNormalizationOfTheWrongWidthOrANonPositiveStd()
    {
        string wide = Json(OneStats[..^1] + ",1]");
        string flat = Json(ZeroStats);

        Assert.Throws<ArgumentException>(() => Trellis2PipelineConfig.FromJson(wide));
        Assert.Throws<ArgumentException>(() => Trellis2PipelineConfig.FromJson(flat));
    }

    [Fact]
    public void ModelFiles_FindTheTextureFilesWhenPresentAndReportShapeOnlyWhenNot()
    {
        string root = Path.Combine(Path.GetTempPath(), "trellis2-tex-" + Guid.NewGuid().ToString("N"));
        string flows = Path.Combine(root, "3D", "trellis2");
        Directory.CreateDirectory(flows);
        try
        {
            foreach (string name in new[]
            {
                Trellis2ModelFiles.PipelineFile, Trellis2ModelFiles.StructureFlowFile, Trellis2ModelFiles.StructureDecoderFile,
                Trellis2ModelFiles.ShapeFlow512File, Trellis2ModelFiles.ShapeFlow1024File, Trellis2ModelFiles.ShapeDecoderFile,
            })
                File.WriteAllBytes(Path.Combine(flows, name), [0]);
            Directory.CreateDirectory(Path.Combine(root, "DINOv3"));
            File.WriteAllBytes(Path.Combine(root, "DINOv3", Trellis2ModelFiles.DinoFile), [0]);

            Trellis2ModelFiles shapeOnly = Trellis2ModelFiles.Resolve(flows);
            Assert.False(shapeOnly.HasTexture);
            Assert.Null(shapeOnly.TextureDecoder);

            File.WriteAllBytes(Path.Combine(flows, Trellis2ModelFiles.TextureDecoderFile), [0]);
            Assert.False(Trellis2ModelFiles.Resolve(flows).HasTexture);

            File.WriteAllBytes(Path.Combine(flows, Trellis2ModelFiles.TextureFlow1024File), [0]);
            Trellis2ModelFiles textured = Trellis2ModelFiles.Resolve(flows);
            Assert.True(textured.HasTexture);
            Assert.EndsWith(Trellis2ModelFiles.TextureFlow1024File, textured.TextureFlow1024, StringComparison.Ordinal);
            Assert.Null(textured.TextureFlow512);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private sealed class Rig : IDisposable
    {
        internal Rig(bool texture = true, bool supports1024 = true)
        {
            Backend = new CpuBackend();
            Owned = new OwnedStub();
            Pipeline = Build(
                Backend,
                new StubDecoder(this),
                texture ? new StubTextureFlow(this, supports1024) : null,
                texture ? new StubTextureDecoder(this) : null);
        }

        internal CpuBackend Backend { get; }

        internal OwnedStub Owned { get; }

        internal Trellis2ImageTo3DPipeline Pipeline { get; }

        internal List<string> Calls { get; } = [];

        internal List<Tensor> Tensors { get; } = [];

        internal Dictionary<int, Tensor> CondByResolution { get; } = [];

        internal Tensor? TextureCond { get; set; }

        internal Trellis2ImageTo3DPipeline Build(
            IBackend backend, ISparseShapeDecoder? decoder = null, ITrellis2TextureFlowStage? textureFlow = null,
            ITrellis2TextureDecoderStage? textureDecoder = null)
            => new(
                backend,
                Config,
                null,
                new StubConditioner(this),
                new StubStructure(this),
                new StubShape(this),
                new StubUpsampler(this),
                decoder ?? new StubDecoder(this),
                new DualGridMesherStage(),
                [Owned],
                textureFlow: textureFlow,
                textureDecoder: textureDecoder);

        public void Dispose()
        {
            Pipeline.Dispose();
            Backend.Dispose();
        }
    }

    private sealed class OwnedStub : IDisposable
    {
        public void Dispose()
        {
        }
    }

    private sealed class StubConditioner(Rig rig) : ITrellis2Conditioner
    {
        public Tensor Encode(IBackend backend, Trellis2PreparedImage image, int resolution)
        {
            rig.Calls.Add($"cond@{resolution}");
            Tensor cond = new(new TensorShape(1, 3, 4), DType.F32);
            rig.Tensors.Add(cond);
            rig.CondByResolution[resolution] = cond;
            return cond;
        }
    }

    private sealed class StubStructure(Rig rig) : ITrellis2StructureStage
    {
        public int[] Sample(
            IBackend backend, Tensor cond, Tensor negCond, int seed, Trellis2SamplerParams parameters, int outputResolution, Action<int>? onStep)
        {
            rig.Calls.Add($"structure@{outputResolution} steps={parameters.Steps} seed={seed}");
            rig.Tensors.Add(negCond);
            for (int s = 1; s <= parameters.Steps; s++)
                onStep?.Invoke(s);
            return [0, 1, 1, 1, 0, 2, 2, 2];
        }
    }

    private sealed class StubShape(Rig rig) : ITrellis2ShapeFlowStage
    {
        public SparseTensor Sample(
            IBackend backend, Trellis2ShapeFlowKind kind, int[] coords, int gridResolution, Tensor cond, Tensor negCond, int seed,
            Trellis2SamplerParams parameters, Action<int>? onStep)
        {
            rig.Calls.Add($"shape {kind} grid={gridResolution} seed={seed} steps={parameters.Steps}");
            for (int s = 1; s <= parameters.Steps; s++)
                onStep?.Invoke(s);
            Tensor latent = new(new TensorShape(1, coords.Length / 4, 32), DType.F32);
            rig.Tensors.Add(latent);
            return new SparseTensor(latent, coords, gridResolution);
        }
    }

    private sealed class StubUpsampler(Rig rig) : ISparseShapeUpsampler
    {
        public int[] Upsample(IBackend backend, SparseTensor latent, int times)
        {
            rig.Calls.Add($"upsample x{times}");
            return [0, 16, 16, 16, 0, 17, 16, 16, 0, 40, 40, 40];
        }
    }

    // One quad (two triangles) around an x-axis edge whose four voxels are the ones the texture stub also fills.
    private static DualGridField Quad()
    {
        int[] coords = [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0];
        float[] dual = new float[12];
        for (int i = 0; i < 12; i++)
            dual[i] = coords[i] + 0.5f;
        bool[] intersected = new bool[12];
        intersected[0] = true;
        return new DualGridField(new DualGridVoxels(coords, dual, intersected), [1f, 1f, 1f, 1f]);
    }

    private sealed class PlainDecoder : ISparseShapeDecoder
    {
        public DualGridField Decode(IBackend backend, SparseTensor latent) => Quad();
    }

    private sealed class StubDecoder(Rig rig) : ISparseShapeSubdivisionDecoder
    {
        public DualGridField Decode(IBackend backend, SparseTensor latent)
        {
            rig.Calls.Add("decode");
            return Quad();
        }

        public DualGridField Decode(IBackend backend, SparseTensor latent, ShapeVaeSubdivisions record)
        {
            rig.Calls.Add("decode+subs");
            Tensor level = new(new TensorShape(8), DType.F32);
            rig.Tensors.Add(level);
            record.Add(level);
            return Quad();
        }
    }

    private sealed class StubTextureFlow(Rig rig, bool supports1024) : ITrellis2TextureFlowStage
    {
        public bool Supports(Trellis2ShapeFlowKind kind) => kind == Trellis2ShapeFlowKind.Res512 || supports1024;

        public SparseTensor Sample(
            IBackend backend, Trellis2ShapeFlowKind kind, SparseTensor shape, Tensor cond, Tensor negCond, int seed,
            Trellis2SamplerParams parameters, Action<int>? onStep)
        {
            rig.Calls.Add($"texture {kind} voxels={shape.Count} seed={seed} steps={parameters.Steps}");
            rig.TextureCond = cond;
            for (int s = 1; s <= parameters.Steps; s++)
                onStep?.Invoke(s);
            Tensor latent = new(new TensorShape(1, shape.Count, 32), DType.F32);
            rig.Tensors.Add(latent);
            return new SparseTensor(latent, shape.Coords, shape.Resolution);
        }
    }

    private sealed class StubTextureDecoder(Rig rig) : ITrellis2TextureDecoderStage
    {
        public SparseTensor Decode(IBackend backend, SparseTensor textureLatent, ShapeVaeSubdivisions guides)
        {
            rig.Calls.Add($"texture-decode levels={guides.Count}");
            // Color (0.8, 0.4, 0.2), metallic 0.25, roughness 0.75, alpha 1 on the four voxels of the stub quad, on the finest grid.
            int[] coords = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0];
            Tensor attributes = new(new TensorShape(1, 4, 6), DType.F32);
            for (int i = 0; i < 4; i++)
                new float[] { 0.8f, 0.4f, 0.2f, 0.25f, 0.75f, 1f }.CopyTo(attributes.AsSpan<float>().Slice(i * 6, 6));
            rig.Tensors.Add(attributes);
            return new SparseTensor(attributes, coords, textureLatent.Resolution * Trellis2Cascade.LatentStride);
        }
    }
}
