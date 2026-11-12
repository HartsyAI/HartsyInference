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
using HartsyInference.ThreeD.Models.Pixal3D;
using HartsyInference.ThreeD.Models.TrellisV2;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;
using HartsyInference.ThreeD.Pipelines.Stages;
using HartsyInference.ThreeD.Tests.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Pixal3D;

/// <summary>The Pixal3D texture path over stub stages: stage order, seeds and conditioning of the texture stage, request checks before any stage runs,
/// progress, cancellation with tensor cleanup, the baked material, and the export frame of the result and its GLB.</summary>
public sealed class Pixal3DTexturePipelineTests
{
    private static readonly string ZeroStats = "[" + string.Join(",", Enumerable.Repeat("0", 32)) + "]";
    private static readonly string OneStats = "[" + string.Join(",", Enumerable.Repeat("1", 32)) + "]";

    private static readonly string Json = $$$"""
        {"name": "Trellis2ImageTo3DPipeline", "args": {
          "sparse_structure_sampler": {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5},
            "params": {"steps": 3, "guidance_strength": 7.5, "guidance_rescale": 0.7, "guidance_interval": [0.6, 1.0], "rescale_t": 5.0}},
          "shape_slat_sampler": {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5},
            "params": {"steps": 2, "guidance_strength": 7.5, "guidance_rescale": 0.5, "guidance_interval": [0.6, 1.0], "rescale_t": 3.0}},
          "shape_slat_normalization": {"mean": {{{ZeroStats}}}, "std": {{{OneStats}}}},
          "tex_slat_sampler": {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5},
            "params": {"steps": 2, "guidance_strength": 1.0, "guidance_rescale": 0.0, "guidance_interval": [0.6, 0.9], "rescale_t": 3.0}},
          "tex_slat_normalization": {"mean": {{{ZeroStats}}}, "std": {{{OneStats}}}} } }
        """;

    private static Trellis2PipelineConfig Config => Trellis2PipelineConfig.FromJson(Json);

    private static Trellis2InputImage Image() => new()
    {
        Rgb = new byte[8 * 8 * 3],
        Alpha = Enumerable.Range(0, 64).Select(i => i is >= 18 and <= 45 ? (byte)255 : (byte)0).ToArray(),
        Width = 8,
        Height = 8,
    };

    private static Pixal3DOptions Options(int seed = 100) => new() { Seed = seed, Texture = true, TextureSize = 16 };

    [Fact]
    public void GenerateTextured_RunsTheTextureStagesAfterTheShapeDecodeOnTheFineVoxels()
    {
        using Rig rig = new();

        rig.Pipeline.GenerateTextured(Image(), Options());

        Assert.Equal(
            [
                "cond image=512 grid=16 dense naf=0",
                "structure@32 steps=3 seed=100",
                "cond image=512 grid=32 voxels=2 naf=512",
                "shape Res512 grid=32 seed=101 steps=2",
                "upsample x4",
                "cond image=1024 grid=64 voxels=2 naf=512",
                "shape Res1024 grid=64 seed=102 steps=2",
                "decode+subs",
                "cond image=1024 grid=64 voxels=2 naf=1024",
                "texture voxels=2 seed=103 steps=2",
                "texture-decode levels=1",
            ],
            rig.Calls);
    }

    [Fact]
    public void GenerateTextured_StepsOverrideTheTextureSamplerToo()
    {
        using Rig rig = new();

        rig.Pipeline.GenerateTextured(Image(), Options() with { Steps = 5 });

        Assert.Contains("texture voxels=2 seed=103 steps=5", rig.Calls);
    }

    [Fact]
    public void Generate_StaysShapeOnlyOnAPipelineWithTextureStages()
    {
        using Rig rig = new();

        Mesh mesh = rig.Pipeline.Generate(Image(), Options());

        Assert.Equal(2, mesh.TriangleCount);
        Assert.Contains("decode", rig.Calls);
        Assert.DoesNotContain(rig.Calls, c => c.StartsWith("texture", StringComparison.Ordinal));
        Assert.DoesNotContain("decode+subs", rig.Calls);
    }

    [Fact]
    public void GenerateTextured_WithoutTextureStagesFailsBeforeAnyStageRuns()
    {
        using Rig rig = new(texture: false);

        Assert.False(rig.Pipeline.HasTexture);
        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => rig.Pipeline.GenerateTextured(Image(), Options()));
        Assert.Contains("texture", error.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Empty(rig.Calls);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(15)]
    [InlineData(8193)]
    public void GenerateTextured_RejectsABadTextureSizeBeforeAnyStageRuns(int size)
    {
        using Rig rig = new();

        Assert.Throws<ArgumentException>(() => rig.Pipeline.GenerateTextured(Image(), Options() with { TextureSize = size }));
        Assert.Throws<ArgumentException>(() => rig.Pipeline.GenerateTextured(Image(), Options() with { DecimationTarget = -1 }));
        Assert.Empty(rig.Calls);
    }

    [Fact]
    public void Constructor_RequiresBothTextureStagesAndASubdivisionDecoder()
    {
        using Rig rig = new();
        Assert.Throws<ArgumentException>(() => rig.Build(rig.Backend, new StubDecoder(rig), new StubTextureFlow(rig), null));
        Assert.Throws<ArgumentException>(() => rig.Build(rig.Backend, new StubDecoder(rig), null, new StubTextureDecoder(rig)));
        Assert.Throws<ArgumentException>(() => rig.Build(rig.Backend, new PlainDecoder(), new StubTextureFlow(rig), new StubTextureDecoder(rig)));
    }

    [Fact]
    public void GenerateTextured_ReportsMonotonicProgressEndingAtTheTotal()
    {
        using Rig rig = new();
        List<GenerationProgress> reports = [];

        rig.Pipeline.GenerateTextured(Image(), Options(), reports.Add);

        // 3 conditionings + upsample + decode = 5 stage steps, 3 structure + 2 + 2 shape sampler steps, then texture: conditioning, 2 steps, decode, bake.
        int total = 5 + 3 + 2 + 2 + 1 + 2 + 2;
        Assert.All(reports, p => Assert.Equal(total, p.TotalSteps));
        Assert.Equal(Enumerable.Range(1, total), reports.Select(p => p.Step));
    }

    // A cancel raised from the progress callback at any step must stop the run and release every tensor a stage handed out.
    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(11)]
    [InlineData(13)]
    [InlineData(14)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(17)]
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
    public void GenerateTextured_ObservesAnAlreadyCancelledTokenBeforeAnyStageRuns()
    {
        using Rig rig = new();
        using CancellationTokenSource cancelled = new();
        cancelled.Cancel();

        Assert.Throws<OperationCanceledException>(() => rig.Pipeline.GenerateTextured(Image(), Options(), null, cancelled.Token));
        Assert.Empty(rig.Calls);
    }

    // The token alone, with no throwing progress callback, must stop the run at the next stage or sampler step and release every tensor.
    [Theory]
    [InlineData(3)]
    [InlineData(9)]
    [InlineData(14)]
    public void GenerateTextured_StopsOnATokenCancelledMidRunAndReleasesEveryTensor(int cancelAtStep)
    {
        using Rig rig = new();
        using CancellationTokenSource source = new();

        Assert.Throws<OperationCanceledException>(() => rig.Pipeline.GenerateTextured(
            Image(),
            Options(),
            p =>
            {
                if (p.Step == cancelAtStep)
                    source.Cancel();
            },
            source.Token));

        Assert.DoesNotContain(rig.Calls, c => c.StartsWith("texture-decode", StringComparison.Ordinal));
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
    public void GenerateTextured_ReturnsTheMeshInTheExportFrameAndToGlbWritesItUnchanged()
    {
        using Rig rig = new();

        PbrPostProcessResult result = rig.Pipeline.GenerateTextured(Image(), Options());

        // The stub quad sits at the grid's -x, -y, -z corner (about -0.5 on every axis): the export frame negates x and z and keeps y.
        float[] v = result.Mesh.Vertices;
        for (int i = 0; i < v.Length; i += 3)
        {
            Assert.InRange(v[i], 0.45f, 0.51f);
            Assert.InRange(v[i + 1], -0.51f, -0.45f);
            Assert.InRange(v[i + 2], 0.45f, 0.51f);
        }

        byte[] glb = Pixal3DImageTo3DPipeline.ToGlb(result);
        Assert.Equal("glTF", System.Text.Encoding.ASCII.GetString(glb, 0, 4));
        Mesh read = GlbReader.Parse(glb);
        Assert.Equal(result.Mesh.TriangleCount, read.TriangleCount);
        Assert.Equal(result.Mesh.Vertices[0], read.Vertices[0], 5);
        Assert.Equal(result.Mesh.Vertices[1], read.Vertices[1], 5);
    }

    private sealed class Rig : IDisposable
    {
        internal Rig(bool texture = true)
        {
            Backend = new CpuBackend();
            Owned = new OwnedStub();
            Pipeline = Build(Backend, new StubDecoder(this), texture ? new StubTextureFlow(this) : null, texture ? new StubTextureDecoder(this) : null);
        }

        internal CpuBackend Backend { get; }

        internal OwnedStub Owned { get; }

        internal Pixal3DImageTo3DPipeline Pipeline { get; }

        internal List<string> Calls { get; } = [];

        internal List<Tensor> Tensors { get; } = [];

        internal Pixal3DImageTo3DPipeline Build(
            IBackend backend, ISparseShapeDecoder decoder, IPixal3DTextureFlowStage? textureFlow, ITrellis2TextureDecoderStage? textureDecoder)
            => new(
                backend,
                Config,
                null,
                new StubConditioner(this),
                new StubStructure(this),
                new StubShape(this),
                new StubUpsampler(this),
                decoder,
                new DualGridMesherStage(),
                [Owned],
                Trellis2Cascade.DefaultMaxTokens,
                textureFlow,
                textureDecoder);

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

    private sealed class StubConditioner(Rig rig) : IPixal3DConditioner
    {
        public Pixal3DCondition Encode(IBackend backend, Trellis2PreparedImage image, Pixal3DConditionSpec spec, Pixal3DView view)
        {
            string where = spec.Coords is null ? "dense" : $"voxels={spec.Coords.Length / 4}";
            rig.Calls.Add($"cond image={spec.ImageSize} grid={spec.GridResolution} {where} naf={spec.NafSize}");
            Tensor global = new(new TensorShape(1, 5, 4), DType.F32);
            Tensor proj = new(new TensorShape(1, 3, 4), DType.F32);
            rig.Tensors.Add(global);
            rig.Tensors.Add(proj);
            return new Pixal3DCondition(global, proj);
        }
    }

    private sealed class StubStructure(Rig rig) : IPixal3DStructureStage
    {
        public int[] Sample(
            IBackend backend, Pixal3DCondition cond, Pixal3DCondition negCond, int seed, Trellis2SamplerParams parameters, int outputResolution,
            Action<int>? onStep)
        {
            rig.Calls.Add($"structure@{outputResolution} steps={parameters.Steps} seed={seed}");
            rig.Tensors.Add(negCond.Global);
            rig.Tensors.Add(negCond.Proj);
            for (int s = 1; s <= parameters.Steps; s++)
                onStep?.Invoke(s);
            return [0, 1, 1, 1, 0, 2, 2, 2];
        }
    }

    private sealed class StubShape(Rig rig) : IPixal3DShapeFlowStage
    {
        public SparseTensor Sample(
            IBackend backend, Trellis2ShapeFlowKind kind, int[] coords, int gridResolution, Pixal3DCondition cond, Pixal3DCondition negCond,
            int seed, Trellis2SamplerParams parameters, Action<int>? onStep)
        {
            rig.Calls.Add($"shape {kind} grid={gridResolution} seed={seed} steps={parameters.Steps}");
            rig.Tensors.Add(negCond.Global);
            rig.Tensors.Add(negCond.Proj);
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
            return [0, 16, 16, 16, 0, 17, 16, 16, 0, 40, 40, 40, 0, 41, 40, 40];
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
            record.Add(level, [0, 0, 0, 0]);
            return Quad();
        }
    }

    private sealed class StubTextureFlow(Rig rig) : IPixal3DTextureFlowStage
    {
        public SparseTensor Sample(
            IBackend backend, SparseTensor shape, Pixal3DCondition cond, Pixal3DCondition negCond, int seed, Trellis2SamplerParams parameters,
            Action<int>? onStep)
        {
            rig.Calls.Add($"texture voxels={shape.Count} seed={seed} steps={parameters.Steps}");
            rig.Tensors.Add(negCond.Global);
            rig.Tensors.Add(negCond.Proj);
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
