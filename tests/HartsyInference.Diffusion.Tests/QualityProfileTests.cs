using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Quality;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

public sealed class QualityProfileTests
{
    [Fact]
    public void From_Maximum_AllFp16()
    {
        QualityProfile p = QualityProfile.From(QualityPreset.Maximum);
        Assert.Equal(DType.F16, p.BackboneDType);
        Assert.Equal(DType.F16, p.TextEncoderDType);
        Assert.Equal(DType.F16, p.VaeDType);
    }

    [Fact]
    public void From_High_Fp8BackboneFp16Encoders()
    {
        QualityProfile p = QualityProfile.From(QualityPreset.High);
        Assert.Equal(DType.F8E4M3, p.BackboneDType);
        Assert.Equal(DType.F16, p.TextEncoderDType);
        Assert.Equal(DType.F16, p.VaeDType);
    }

    [Fact]
    public void From_Custom_Throws()
    {
        Assert.Throws<ArgumentException>(() => QualityProfile.From(QualityPreset.Custom));
    }

    [Fact]
    public void Validate_RejectsFp8Vae()
    {
        QualityProfile p = new QualityProfile { BackboneDType = DType.F16, TextEncoderDType = DType.F16, VaeDType = DType.F8E4M3 };
        Assert.Throws<HartsyInferenceException>(() => p.Validate());
    }

    [Fact]
    public void Validate_RejectsQuantizedVae()
    {
        QualityProfile p = new QualityProfile { BackboneDType = DType.F16, TextEncoderDType = DType.F16, VaeDType = DType.Q8_0 };
        Assert.Throws<HartsyInferenceException>(() => p.Validate());
    }

    [Fact]
    public void Apply_F32ToF16_CastsRank2Weights()
    {
        Dictionary<string, Tensor> weights = new()
        {
            ["transformer.linear_1.weight"] = MakeF32Tensor(new TensorShape(64, 32)),
            ["transformer.linear_1.bias"] = MakeF32Tensor(new TensorShape(64)),
            ["transformer.norm.weight"] = MakeF32Tensor(new TensorShape(64)),
        };
        try
        {
            int n = QualityProfileApplier.Apply(weights, DType.F16);
            Assert.Equal(3, n);
            Assert.Equal(DType.F16, weights["transformer.linear_1.weight"].DType);
            Assert.Equal(DType.F16, weights["transformer.linear_1.bias"].DType);
            Assert.Equal(DType.F16, weights["transformer.norm.weight"].DType);
        }
        finally
        {
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }

    [Fact]
    public void Apply_F32ToFp8_SkipsNormsAndBiases()
    {
        Dictionary<string, Tensor> weights = new()
        {
            ["transformer.linear_1.weight"] = MakeF32Tensor(new TensorShape(64, 32)),
            ["transformer.linear_1.bias"] = MakeF32Tensor(new TensorShape(64)),
            ["transformer.norm.weight"] = MakeF32Tensor(new TensorShape(64)),
            ["transformer.layernorm.weight"] = MakeF32Tensor(new TensorShape(64)),
            ["transformer.pos_embed.pos_embed"] = MakeF32Tensor(new TensorShape(1, 1024, 32)),
        };
        try
        {
            QualityProfileApplier.Apply(weights, DType.F8E4M3);
            Assert.Equal(DType.F8E4M3, weights["transformer.linear_1.weight"].DType);
            Assert.Equal(DType.F32, weights["transformer.linear_1.bias"].DType);
            Assert.Equal(DType.F32, weights["transformer.norm.weight"].DType);
            Assert.Equal(DType.F32, weights["transformer.layernorm.weight"].DType);
            Assert.Equal(DType.F32, weights["transformer.pos_embed.pos_embed"].DType);
        }
        finally
        {
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }

    [Fact]
    public void Apply_QuantizedTarget_NoOps()
    {
        Dictionary<string, Tensor> weights = new()
        {
            ["w"] = MakeF32Tensor(new TensorShape(8, 8)),
        };
        try
        {
            int n = QualityProfileApplier.Apply(weights, DType.Q8_0);
            Assert.Equal(0, n);
            Assert.Equal(DType.F32, weights["w"].DType);
        }
        finally
        {
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }

    [Fact]
    public void FluxQualityLoader_AppliesPerComponentDtypes()
    {
        HartsyInference.ModelAssets.CheckpointConverters.FluxCheckpointConverter.ConvertedWeights converted = new()
        {
            Transformer = new() { ["t.linear.weight"] = MakeF32Tensor(new TensorShape(64, 32)) },
            ClipL = new() { ["c.linear.weight"] = MakeF32Tensor(new TensorShape(32, 16)) },
            T5 = new() { ["t5.linear.weight"] = MakeF32Tensor(new TensorShape(32, 16)) },
            Vae = new() { ["v.linear.weight"] = MakeF32Tensor(new TensorShape(16, 8)) },
        };

        try
        {
            QualityProfile high = QualityProfile.From(QualityPreset.High);
            FluxQualityLoader.Apply(converted, high);
            Assert.Equal(DType.F8E4M3, converted.Transformer["t.linear.weight"].DType);
            Assert.Equal(DType.F16, converted.ClipL["c.linear.weight"].DType);
            Assert.Equal(DType.F16, converted.T5["t5.linear.weight"].DType);
            Assert.Equal(DType.F16, converted.Vae["v.linear.weight"].DType);
        }
        finally
        {
            foreach (Tensor t in converted.Transformer.Values) t.Dispose();
            foreach (Tensor t in converted.ClipL.Values) t.Dispose();
            foreach (Tensor t in converted.T5.Values) t.Dispose();
            foreach (Tensor t in converted.Vae.Values) t.Dispose();
        }
    }

    [Fact]
    public void SdxlQualityLoader_AppliesPerComponentDtypes()
    {
        HartsyInference.ModelAssets.CheckpointConverters.SdxlCheckpointConverter.ConvertedWeights converted = new()
        {
            UNet = new() { ["u.linear.weight"] = MakeF32Tensor(new TensorShape(64, 32)) },
            ClipL = new() { ["cl.linear.weight"] = MakeF32Tensor(new TensorShape(32, 16)) },
            ClipG = new() { ["cg.linear.weight"] = MakeF32Tensor(new TensorShape(32, 16)) },
            Vae = new() { ["v.linear.weight"] = MakeF32Tensor(new TensorShape(16, 8)) },
        };

        try
        {
            QualityProfile maximum = QualityProfile.From(QualityPreset.Maximum);
            SdxlQualityLoader.Apply(converted, maximum);
            Assert.Equal(DType.F16, converted.UNet["u.linear.weight"].DType);
            Assert.Equal(DType.F16, converted.ClipL["cl.linear.weight"].DType);
            Assert.Equal(DType.F16, converted.ClipG["cg.linear.weight"].DType);
            Assert.Equal(DType.F16, converted.Vae["v.linear.weight"].DType);
        }
        finally
        {
            foreach (Tensor t in converted.UNet.Values) t.Dispose();
            foreach (Tensor t in converted.ClipL.Values) t.Dispose();
            foreach (Tensor t in converted.ClipG.Values) t.Dispose();
            foreach (Tensor t in converted.Vae.Values) t.Dispose();
        }
    }

    [Fact]
    public void FluxQualityLoader_RejectsInvalidVaeDtype()
    {
        HartsyInference.ModelAssets.CheckpointConverters.FluxCheckpointConverter.ConvertedWeights converted = new()
        {
            Transformer = new(),
            ClipL = new(),
            T5 = new(),
            Vae = new(),
        };
        QualityProfile bad = new() { BackboneDType = DType.F16, TextEncoderDType = DType.F16, VaeDType = DType.F8E4M3 };
        Assert.Throws<HartsyInferenceException>(() => FluxQualityLoader.Apply(converted, bad));
    }

    private static unsafe Tensor MakeF32Tensor(TensorShape shape)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        long n = shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = 1.0f;
        return t;
    }
}
