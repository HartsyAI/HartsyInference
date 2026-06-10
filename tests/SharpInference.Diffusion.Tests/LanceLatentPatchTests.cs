using Xunit;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace SharpInference.Diffusion.Tests;

/// <summary>Tests for the Lance VAE↔transformer latent handoff (<see cref="LanceLatentPatch"/>), the <c>(1,2,2)</c> pixel-shuffle between the channel-last latent and the 192-dim token sequence. No GPU/checkpoint.</summary>
public unsafe class LanceLatentPatchTests
{
    [Fact]
    public void PatchifyUnpatchify_RoundTrips()
    {
        int t = 1, h = 3, w = 4, c = 48, pt = 1, ph = 2, pw = 2;
        Tensor latent = Random([t * pt, h * ph, w * pw, c], seed: 7);
        Tensor tokens = LanceLatentPatch.Patchify(latent, pt, ph, pw);

        Assert.Equal(t * h * w, (int)tokens.Shape[0]);
        Assert.Equal(pt * ph * pw * c, (int)tokens.Shape[1]); // 192

        Tensor back = LanceLatentPatch.Unpatchify(tokens, t, h, w, pt, ph, pw, c);
        AssertEqual(latent, back);
    }

    [Fact]
    public void Patchify_FeatureOrderMatchesUpstreamEinops()
    {
        // (pt ph pw c) feature index = ((pti*ph+phi)*pw+pwi)*c + ci. For pt=1,ph=pw=2,c=2:
        // a single (h=0,w=0) block's 4 spatial cells map to feature blocks [0:c],[c:2c],[2c:3c],[3c:4c].
        int c = 2, ph = 2, pw = 2;
        Tensor latent = new Tensor(new TensorShape([1L, 2, 2, c]), DType.F32); // [Tf=1,Hf=2,Wf=2,C=2]
        float* p = (float*)latent.DataPointer;
        // value = 100*hf + 10*wf + ci  (channel-last index ((hf)*Wf+wf)*c+ci)
        for (int hf = 0; hf < 2; hf++)
            for (int wf = 0; wf < 2; wf++)
                for (int ci = 0; ci < c; ci++)
                    p[(hf * 2 + wf) * c + ci] = 100 * hf + 10 * wf + ci;

        Tensor tokens = LanceLatentPatch.Patchify(latent, 1, ph, pw); // [1, 8]
        float* tk = (float*)tokens.DataPointer;
        // feature = ((phi)*pw+pwi)*c + ci
        for (int phi = 0; phi < 2; phi++)
            for (int pwi = 0; pwi < 2; pwi++)
                for (int ci = 0; ci < c; ci++)
                {
                    int feat = (phi * pw + pwi) * c + ci;
                    Assert.Equal(100 * phi + 10 * pwi + ci, tk[feat]);
                }
    }

    private static Tensor Random(int[] dims, int seed)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new Tensor(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static void AssertEqual(Tensor a, Tensor b)
    {
        long n = a.Shape.ElementCount;
        float* pa = (float*)a.DataPointer;
        float* pb = (float*)b.DataPointer;
        for (long i = 0; i < n; i++) Assert.Equal(pa[i], pb[i]);
    }
}
