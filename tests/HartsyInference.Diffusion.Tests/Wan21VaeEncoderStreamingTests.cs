using Xunit;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The Wan VAE encode was one contiguous conv workspace over the whole clip — 5.8 GB for a 41-frame
/// 480x800 driving clip, which is what put the official Wan-Animate buckets out of reach beside a resident DiT.
/// It now streams on the reference's chunking (frame 0 alone, then groups of 4). These pin the streamed result to
/// the whole-clip result it replaced: a streaming migration that changes numerics is a regression, not a win.</summary>
public unsafe class Wan21VaeEncoderStreamingTests
{
    private static Tensor Rgb(int t, int h, int w, int seed)
    {
        Tensor x = new Tensor(new TensorShape([1L, 3, t, h, w]), DType.F32);
        Random rng = new Random(seed);
        Span<float> d = new Span<float>((float*)x.DataPointer, checked((int)x.Shape.ElementCount));
        for (int i = 0; i < d.Length; i++) d[i] = (float)(rng.NextDouble() * 2 - 1);
        return x;
    }

    private static Wan21VaeEncoder Build()
    {
        Wan21VaeEncoder enc = new Wan21VaeEncoder(
            dim: 8, zDim: 16, dimMult: [1, 2, 2], numResBlocks: 1, temperalDownsample: [true, true]);
        enc.LoadWeights(MatrixGame2SyntheticWeights.BuildVae21Encoder(
            dim: 8, zDim: 16, dimMult: [1, 2, 2], numResBlocks: 1, tDown: [true, true]));
        return enc;
    }

    [Theory]
    [InlineData(5)]
    [InlineData(9)]
    [InlineData(13)]
    public void StreamedEncode_MatchesTheWholeClipEncode(int frames)
    {
        using CpuBackend backend = new CpuBackend();
        Wan21VaeEncoder enc = Build();
        using Tensor rgb = Rgb(frames, 16, 16, seed: 5150 + frames);

        Tensor streamed, whole;
        try
        {
            Wan21VaeEncoder.ForceWholeClipEncode = true;
            whole = enc.Encode(backend, rgb);
        }
        finally { Wan21VaeEncoder.ForceWholeClipEncode = false; }
        streamed = enc.Encode(backend, rgb);

        using (streamed)
        using (whole)
        {
            Assert.Equal(whole.Shape.ToString(), streamed.Shape.ToString());
            Assert.Equal(1 + (frames - 1) / 4, (int)streamed.Shape[2]);
            float* a = (float*)whole.DataPointer, b = (float*)streamed.DataPointer;
            long n = whole.Shape.ElementCount;
            double worst = 0;
            for (long i = 0; i < n; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
            Assert.True(worst < 2e-4, $"streamed encode diverged from the whole-clip encode by {worst}");
        }
    }
}
