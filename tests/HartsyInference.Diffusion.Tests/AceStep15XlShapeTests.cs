using System;
using System.Collections.Generic;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Shape/plumbing test for the XL geometry: a wider decoder over a NARROWER condition encoder
/// (real XL: enc 2048/16H under dec 2560/32H with q = 32·128 = 4096 ≠ hidden — rectangular attention).
/// Scaled-down here (enc 16/2H, dec 24/4H·8hd = q 32 ≠ 24) so a CPU forward runs in milliseconds; the real
/// checkpoint path reuses the parity-verified 2B code with <see cref="AceStep15Config.FromJson"/> dims.</summary>
public sealed unsafe class AceStep15XlShapeTests
{
    private static AceStep15Config TinyXl => AceStep15SyntheticWeights.TinyConfig with
    {
        HiddenSize = 24, NumHeads = 4, NumKvHeads = 2, HeadDim = 8, IntermediateSize = 48,
        EncoderHiddenSize = 16, EncoderIntermediateSize = 32, EncoderNumHeads = 2, EncoderNumKvHeads = 1,
        IsTurbo = false,
    };

    [Fact]
    public void XlGeometry_EncoderAndDit_ForwardShapes()
    {
        AceStep15Config cfg = TinyXl;
        Dictionary<string, Tensor> w = AceStep15SyntheticWeights.BuildModel(cfg);
        using CpuBackend backend = new();

        AceStep15ConditionEncoder encoder = new(cfg.EncoderVariant());
        encoder.LoadWeights(w);
        Tensor text = Rand(1, 5, cfg.TextHiddenDim);
        Tensor lyric = Rand(1, 7, cfg.TextHiddenDim);
        Tensor conditions = encoder.EncodeConditions(backend, text, lyric, null);
        Assert.Equal(cfg.EncoderVariant().HiddenSize, (int)conditions.Shape[2]);   // packed at ENCODER width

        AceStep15Dit dit = new(cfg);
        dit.LoadWeights(w);
        int frames = 8;
        Tensor noisy = Rand(1, frames, cfg.LatentChannels);
        Tensor context = Rand(1, frames, 2 * cfg.LatentChannels);
        Tensor v = dit.Forward(backend, noisy, context, conditions, 0.75f, 0.75f);
        Assert.Equal(frames, (int)v.Shape[1]);
        Assert.Equal(cfg.LatentChannels, (int)v.Shape[2]);
        float* vp = (float*)v.DataPointer;
        for (long i = 0; i < v.Shape.ElementCount; i++)
        {
            Assert.False(float.IsNaN(vp[i]) || float.IsInfinity(vp[i]), $"non-finite velocity at {i}");
        }
        v.Dispose(); conditions.Dispose(); text.Dispose(); lyric.Dispose(); noisy.Dispose(); context.Dispose();
    }

    private static Tensor Rand(int b, int t, int c)
    {
        Tensor outT = new(new TensorShape(b, t, c), DType.F32);
        Random rng = new(7);
        float* p = (float*)outT.DataPointer;
        for (long i = 0; i < outT.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() - 0.5);
        return outT;
    }
}
