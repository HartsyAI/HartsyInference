using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.HairDeck;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>Real-weight parity of the HairDeck core against <c>tools/parity/dump_hairdeck_core.py</c> (CPU, fp32). Every stage
/// gets the reference's inputs (the DINOv3 tokens, the initial noise and the chain noise are injected), so each error is that
/// stage's own. Tolerances are peak-relative: max abs error over the peak |reference|.</summary>
[Trait("Category", "Integration")]
public sealed class HairDeckCoreParityTests : IClassFixture<HairDeckParityComponents>
{
    private const int Slots = 192, TokenDim = 40, DinoTokens = 789, Width = 768, ChainSlots = 48, ChainDim = 27;
    private const double StageTolerance = 1e-4;
    private const double SamplerTolerance = 1e-3;

    private readonly HairDeckParityComponents _c;
    private readonly ITestOutputHelper _output;

    public HairDeckCoreParityTests(HairDeckParityComponents components, ITestOutputHelper output)
    {
        _c = components;
        _output = output;
    }

    [HairDeckRealWeightsFact]
    public void ImageContext_MatchesReference()
    {
        using IBackend backend = new CpuBackend();
        using Tensor dino = HairDeckParityComponents.Make(_c.Dump.Floats("dino_tokens"), DinoTokens, 1024);
        using Tensor ctx = _c.Image.EncodeContext(backend, dino);
        Check("image context", HairDeckNumerics.ToArray(ctx), _c.Dump.Doubles("img_ctx"), StageTolerance);
    }

    [HairDeckRealWeightsFact]
    public void Denoiser_CondAndUncond_MatchReference()
    {
        using IBackend backend = new CpuBackend();
        using Tensor dino = HairDeckParityComponents.Make(_c.Dump.Floats("dino_tokens"), DinoTokens, 1024);
        using Tensor ctx = _c.Image.EncodeContext(backend, dino);
        using Tensor noise = HairDeckParityComponents.Make(_c.Dump.Floats("noise_cards"), Slots, TokenDim);
        float t = _c.Dump.Floats("denoise_t")[0];
        using (Tensor cond = _c.Image.Core.Forward(backend, noise, t, ctx, uncond: false))
            Check("denoiser cond", HairDeckNumerics.ToArray(cond), _c.Dump.Doubles("denoise_cond"), StageTolerance);
        using (Tensor uncond = _c.Image.Core.Forward(backend, noise, t, ctx, uncond: true))
            Check("denoiser uncond", HairDeckNumerics.ToArray(uncond), _c.Dump.Doubles("denoise_uncond"), StageTolerance);
    }

    [HairDeckRealWeightsFact]
    public void Sampler_FiftyStepTokens_MatchReference()
    {
        using IBackend backend = new CpuBackend();
        using Tensor dino = HairDeckParityComponents.Make(_c.Dump.Floats("dino_tokens"), DinoTokens, 1024);
        using Tensor ctx = _c.Image.EncodeContext(backend, dino);
        using Tensor noise = HairDeckParityComponents.Make(_c.Dump.Floats("noise_cards"), Slots, TokenDim);
        using Tensor tokens = HairDeckEulerSampler.Sample(backend, _c.Image.Core, ctx, noise, 50, 2f);
        Check("50-step card tokens", HairDeckNumerics.ToArray(tokens), _c.Dump.Doubles("final_tokens"), SamplerTolerance);
    }

    [HairDeckRealWeightsFact]
    public void CleanAndSignTokens_MatchReference()
    {
        using Tensor tokens = HairDeckParityComponents.Make(_c.Dump.Floats("final_tokens"), Slots, TokenDim);
        using Tensor clean = _c.Tokenizer.Clean(tokens);
        Check("clean tokens", HairDeckNumerics.ToArray(clean), _c.Dump.Doubles("clean_tokens"), StageTolerance);
        using Tensor signed = _c.Tokenizer.SignNormals(clean);
        Check("sign-ruled tokens", HairDeckNumerics.ToArray(signed), _c.Dump.Doubles("sign_tokens"), StageTolerance);
    }

    [HairDeckRealWeightsFact]
    public void CardVae_CountsAndDecodedPoints_MatchReference()
    {
        using IBackend backend = new CpuBackend();
        float[] latents = _c.Dump.Floats("vae_z");
        double[] roots = _c.Dump.Doubles("vae_root");
        long[] counts = _c.Dump.Longs("vae_count");
        double[] points = _c.Dump.Doubles("vae_points");
        List<float> ours = [];
        for (int i = 0; i < counts.Length; i++)
        {
            ReadOnlySpan<float> z = latents.AsSpan(i * 32, 32);
            float[] root = [(float)roots[i * 3], (float)roots[i * 3 + 1], (float)roots[i * 3 + 2]];
            Assert.Equal((int)counts[i], _c.Vae.PointCount(backend, z));
            ours.AddRange(_c.Vae.DecodeCard(backend, z, root));
        }
        Check("card VAE decoded points", ours.ToArray(), points, StageTolerance);
    }

    [HairDeckRealWeightsFact]
    public void ColorHead_ContextAndLab_MatchReference()
    {
        using IBackend backend = new CpuBackend();
        using Tensor dino = HairDeckParityComponents.Make(_c.Dump.Floats("dino_tokens"), DinoTokens, 1024);
        using Tensor image = HairDeckParityComponents.Make(_c.Dump.Floats("color_img"), 3, 448, 448);
        using Tensor matte = HairDeckParityComponents.Make(_c.Dump.Floats("color_matte"), 448, 448);
        using Tensor ctx = _c.Color.EncodeContext(backend, dino, image, matte);
        Check("colour context", HairDeckNumerics.ToArray(ctx), _c.Dump.Doubles("color_ctx"), StageTolerance);
        using Tensor signed = HairDeckParityComponents.Make(_c.Dump.Floats("sign_tokens"), Slots, TokenDim);
        using Tensor lab = _c.Color.PredictLab(backend, signed, ctx);
        Check("colour Lab", HairDeckNumerics.ToArray(lab), _c.Dump.Doubles("color_lab"), StageTolerance);
    }

    [HairDeckRealWeightsFact]
    public void ChainStep_AndSampledChains_MatchReference()
    {
        using IBackend backend = new CpuBackend();
        using Tensor clean = HairDeckParityComponents.Make(_c.Dump.Floats("clean_tokens"), Slots, TokenDim);
        using Tensor ctx = _c.Chain.EncodeContext(backend, clean);
        Check("chain context", HairDeckNumerics.ToArray(ctx), _c.Dump.Doubles("chain_ctx"), StageTolerance);
        using Tensor noise = HairDeckParityComponents.Make(_c.Dump.Floats("chain_noise"), ChainSlots, ChainDim);
        float t = _c.Dump.Floats("denoise_t")[0];
        using (Tensor cond = _c.Chain.Core.Forward(backend, noise, t, ctx, uncond: false))
            Check("chain denoiser cond", HairDeckNumerics.ToArray(cond), _c.Dump.Doubles("chain_denoise_cond"), StageTolerance);
        using (Tensor uncond = _c.Chain.Core.Forward(backend, noise, t, ctx, uncond: true))
            Check("chain denoiser uncond", HairDeckNumerics.ToArray(uncond), _c.Dump.Doubles("chain_denoise_uncond"), StageTolerance);

        using Tensor sampled = HairDeckEulerSampler.Sample(backend, _c.Chain.Core, ctx, noise, 50, 2f);
        Check("50-step chain tokens", HairDeckNumerics.ToArray(sampled), _c.Dump.Doubles("chain_final"), SamplerTolerance);

        IReadOnlyList<DecodedChain> chains = HairDeckChainDecoder.Decode(sampled, _c.ChainMu, _c.ChainSd);
        long[] counts = _c.Dump.Longs("chain_count");
        Assert.Equal(counts.Length, chains.Count);
        for (int i = 0; i < chains.Count; i++) Assert.Equal((int)counts[i], chains[i].JointCount);
        Check("chain joints", chains.SelectMany(ch => ch.Joints).ToArray(), _c.Dump.Doubles("chain_joints"), StageTolerance);
        Check("chain paths", chains.SelectMany(ch => ch.Path).ToArray(), _c.Dump.Doubles("chain_path"), StageTolerance);
        Check("chain stiffness", chains.Select(ch => ch.Stiffness).ToArray(), _c.Dump.Doubles("chain_stiffness"), StageTolerance);
    }

    [HairDeckRealWeightsFact]
    public void EndToEnd_CardsColorsChains_MatchReference()
    {
        using IBackend backend = new CpuBackend();
        HairDeckModelFiles files = new(HairDeckReferenceLayout.ImageModel, HairDeckReferenceLayout.CardVae,
            HairDeckReferenceLayout.CardVaeStats, HairDeckReferenceLayout.TokenStats, HairDeckReferenceLayout.LatentStats,
            HairDeckReferenceLayout.ColorModel, HairDeckReferenceLayout.ChainModel, HairDeckReferenceLayout.ChainStats);
        using HairDeckCore core = HairDeckCore.Load(files);
        using Tensor dino = HairDeckParityComponents.Make(_c.Dump.Floats("dino_tokens"), DinoTokens, 1024);
        using Tensor image = HairDeckParityComponents.Make(_c.Dump.Floats("color_img"), 3, 448, 448);
        using Tensor matte = HairDeckParityComponents.Make(_c.Dump.Floats("color_matte"), 448, 448);
        using Tensor noise = HairDeckParityComponents.Make(_c.Dump.Floats("noise_cards"), Slots, TokenDim);
        using Tensor chainNoise = HairDeckParityComponents.Make(_c.Dump.Floats("chain_noise"), ChainSlots, ChainDim);
        HairDeckResult result = core.GenerateWithNoise(backend, dino, image, matte, noise, chainNoise, steps: 50, cfg: 2f, withChains: true);

        long[] counts = _c.Dump.Longs("card_count");
        long[] shapes = _c.Dump.Longs("card_shape");
        long[] slots = _c.Dump.Longs("card_slot");
        double[] rows = _c.Dump.Doubles("card_points");
        double[] srgb = _c.Dump.Doubles("color_srgb");
        Assert.Equal(counts.Length, result.Cards.Count);

        List<double> xyzRef = [], xyzOurs = [], widthRef = [], widthOurs = [], thickRef = [], thickOurs = [];
        List<double> normalRef = [], normalOurs = [], colorRef = [], colorOurs = [];
        int row = 0;
        for (int i = 0; i < counts.Length; i++)
        {
            DecodedCard card = result.Cards[i];
            Assert.Equal((int)counts[i], card.Points);
            Assert.Equal((int)shapes[i], card.ShapeClass);
            for (int j = 0; j < card.Points; j++, row++)
            {
                for (int c = 0; c < 3; c++)
                {
                    xyzRef.Add(rows[row * 8 + c]);
                    xyzOurs.Add(card.Centerline[j * 3 + c]);
                }
                widthRef.Add(rows[row * 8 + 3]);
                widthOurs.Add(card.WidthProfile[j]);
                thickRef.Add(rows[row * 8 + 4]);
                thickOurs.Add(card.ThicknessProfile[j]);
            }
            for (int c = 0; c < 3; c++)
            {
                normalRef.Add(rows[(row - card.Points) * 8 + 5 + c]);
                normalOurs.Add(card.RootNormal[c]);
            }
            int slot = (int)slots[i];
            for (int k = 0; k < 48; k++)
            {
                colorRef.Add(srgb[slot * 48 + k]);
                colorOurs.Add(LinearToSrgb(card.Color![k]));
            }
        }
        Check("cards centreline", xyzOurs.Select(v => (float)v).ToArray(), xyzRef.ToArray(), StageTolerance);
        Check("cards width", widthOurs.Select(v => (float)v).ToArray(), widthRef.ToArray(), StageTolerance);
        Check("cards thickness", thickOurs.Select(v => (float)v).ToArray(), thickRef.ToArray(), StageTolerance);
        Check("cards root normal", normalOurs.Select(v => (float)v).ToArray(), normalRef.ToArray(), StageTolerance);
        Check("cards colour (sRGB of linear)", colorOurs.Select(v => (float)v).ToArray(), colorRef.ToArray(), StageTolerance);

        long[] chainCounts = _c.Dump.Longs("chain_count");
        double[] chainJoints = _c.Dump.Doubles("chain_joints");
        Assert.Equal(chainCounts.Length, result.Chains.Count);
        Check("end-to-end chain joints", result.Chains.SelectMany(ch => ch.Joints).ToArray(), chainJoints, StageTolerance);
    }

    private void Check(string stage, float[] ours, double[] reference, double tolerance)
    {
        HairDeckError e = HairDeckError.Compare(ours, reference);
        _output.WriteLine($"{stage}: {e} (tolerance {tolerance:G3})");
        Assert.True(e.PeakRelative <= tolerance, $"{stage} exceeds tolerance: {e}");
    }

    private static double LinearToSrgb(float linear) =>
        linear <= 0.0031308f ? 12.92 * linear : 1.055 * Math.Pow(linear, 1.0 / 2.4) - 0.055;
}
