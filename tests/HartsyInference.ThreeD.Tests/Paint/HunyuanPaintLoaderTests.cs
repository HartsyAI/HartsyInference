using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

namespace HartsyInference.ThreeD.Tests.Paint;

/// <summary>Unit lane: generation detection from tensor names and shapes, and the loader's refusals.</summary>
public sealed class HunyuanPaintLoaderTests
{
    private static Tensor Empty(params long[] shape) => new(new TensorShape(shape), DType.F32);

    [Fact]
    public void Detect_ReadsTheGenerationFromTheUnetTensors()
    {
        HunyuanPaintConfig v21 = HunyuanPaintLoader.Detect(new Dictionary<string, Tensor> { ["unet.learned_text_clip_albedo"] = Empty(77, 1024) });
        HunyuanPaintConfig v20 = HunyuanPaintLoader.Detect(new Dictionary<string, Tensor> { ["unet.class_embedding.weight"] = Empty(49, 1280) });

        Assert.Equal(["albedo", "mr"], v21.Materials);
        Assert.True(v21.UseDino && v21.UsePoseRope);
        Assert.Equal(["albedo"], v20.Materials);
        Assert.Equal(49, v20.CameraClasses);
        Assert.False(v20.UseDino);
        Assert.Throws<InvalidDataException>(() => HunyuanPaintLoader.Detect(new Dictionary<string, Tensor>()));
    }

    [Fact]
    public void Load_RefusesAFolderWithoutTheUnet()
    {
        string folder = Path.Combine(Path.GetTempPath(), "paint-empty-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(folder);
        try
        {
            using CpuBackend backend = new();
            Assert.Throws<FileNotFoundException>(() => HunyuanPaintLoader.Load(backend, folder));
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    [Fact]
    public void PromptTokens_ReadThePerMaterialAndReferenceBlocksWithOrWithoutTheGenPrefix()
    {
        HunyuanPaintConfig cfg = HunyuanPaintTestWeights.TinyConfig(v21: false);
        Dictionary<string, Tensor> weights = new()
        {
            ["unet.learned_text_clip_gen"] = Empty(1, cfg.TextTokens, cfg.CrossAttentionDim),
            ["unet.learned_text_clip_ref"] = Empty(1, cfg.TextTokens, cfg.CrossAttentionDim),
        };

        HunyuanPaintPromptTokens tokens = HunyuanPaintPromptTokens.FromWeights(weights, cfg, "unet");

        Assert.Single(tokens.PerMaterial);
        Assert.Equal(new TensorShape(1, cfg.TextTokens, cfg.CrossAttentionDim), tokens.PerMaterial[0].Shape);
        Assert.Equal(tokens.PerMaterial[0].Shape, tokens.Reference.Shape);
    }

    [Fact]
    public unsafe void F32Weights_ConvertHalfPrecisionOnceAndReleaseOnlyTheCopies()
    {
        using Tensor half = Empty(2, 3).CastTo(DType.F16), single = Empty(2);
        HunyuanPaintF32Weights view = new(new Dictionary<string, Tensor> { ["half"] = half, ["single"] = single });

        Tensor converted = view["half"];
        Assert.Equal(DType.F32, converted.DType);
        Assert.Same(converted, view["half"]);
        Assert.Same(single, view["single"]);

        view.Dispose();

        Assert.Equal(DType.F16, half.DType);
        Assert.Equal(0f, *(float*)single.DataPointer);
    }

    [Fact]
    public void StripPrefix_ExposesABundledDinoUnderItsStandaloneNames()
    {
        using Tensor cls = Empty(1, 1, 4), other = Empty(2);
        Dictionary<string, Tensor> bundle = new()
        {
            ["conditioner.main_image_encoder.model.embeddings.cls_token"] = cls,
            ["conditioner.main_image_encoder.model.layernorm.weight"] = other,
            ["model.double_blocks.0.img_mod.lin.weight"] = other,
        };

        IReadOnlyDictionary<string, Tensor> stripped = HunyuanPaintLoader.StripPrefix(bundle, HunyuanPaintLoader.BundledDinoPrefix);

        Assert.Equal(2, stripped.Count);
        Assert.Same(cls, stripped["embeddings.cls_token"]);
        Assert.Same(bundle, HunyuanPaintLoader.StripPrefix(bundle, "absent."));
        Dictionary<string, Tensor> standalone = new() { ["embeddings.cls_token"] = cls };
        Assert.Same(standalone, HunyuanPaintLoader.StripPrefix(standalone, HunyuanPaintLoader.BundledDinoPrefix));
    }
}
