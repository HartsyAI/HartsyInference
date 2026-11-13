using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Hunyuan3D.Paint;
using HartsyInference.ThreeD.Paint;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Paint;

/// <summary>Integration lane: one whole denoising step of Paint 2.1, chained as the pipeline chains it (VAE encodes of the reference, normal and position pictures, the guidance branches through the
/// real UNet, the view-dependent guidance, one UniPC step, the VAE decode) against upstream's own modules on the CPU (<c>tests/python-reference/dump_paint21_step.py</c>). The DINO tokens are the
/// ones Python projected, fed through a stand-in conditioner (the encoder has its own comparison). Set <c>HARTSY_PAINT21_UNET</c> (the UNet pickle inside the model folder) and
/// <c>HARTSY_PAINT21_STEP</c> (the fixture).</summary>
[Trait("Category", "Integration")]
public sealed class HunyuanPaintStepReferenceTests(ITestOutputHelper output)
{
    private sealed class FixedConditioner(float[] conditional, float[] unconditional, int tokens, int width) : IPaintConditioner
    {
        public Tensor Tokens(IBackend backend, ReadOnlySpan<byte> rgb, int imageWidth, int imageHeight) => Copy(conditional);

        public Tensor UnconditionalTokens(IBackend backend) => Copy(unconditional);

        private Tensor Copy(float[] values)
        {
            Tensor t = new(new TensorShape(1, tokens, width), DType.F32);
            values.CopyTo(t.AsSpan<float>());
            return t;
        }
    }

    private static float[] Values(Dictionary<string, Tensor> fixture, string key) => fixture[key].AsSpan<float>().ToArray();

    private static byte[][] Pictures(float[] values, int count, int bytesEach) =>
        [.. Enumerable.Range(0, count).Select(i => values.AsSpan(i * bytesEach, bytesEach).ToArray().Select(v => (byte)Math.Round(v)).ToArray())];

    [RealWeightsFact("HARTSY_PAINT21_UNET", "HARTSY_PAINT21_STEP")]
    public void OneDenoisingStep_MatchesUpstreamThroughEncodeUnetGuidanceSamplerAndDecode()
    {
        string unetFile = Environment.GetEnvironmentVariable("HARTSY_PAINT21_UNET")!;
        string folder = Path.GetDirectoryName(Path.GetDirectoryName(unetFile))!;
        using SafeTensorsLoader fixtureLoader = new();
        fixtureLoader.Load(Environment.GetEnvironmentVariable("HARTSY_PAINT21_STEP")!);
        Dictionary<string, Tensor> fixture = fixtureLoader.GetAllTensors();
        const int size = 128, views = 6, materials = 2, latent = size / 8;
        int bytesEach = size * size * 3;

        float[] cond = Values(fixture, "dino_proj_cond"), uncond = Values(fixture, "dino_proj_uncond");
        int tokens = (int)fixture["dino_proj_cond"].Shape[0], width = (int)fixture["dino_proj_cond"].Shape[1];
        float[] init = Values(fixture, "init_latents");
        int perMaterial = views * 4 * latent * latent;
        float[][] initial = [.. Enumerable.Range(0, materials).Select(m => init.AsSpan(m * perMaterial, perMaterial).ToArray())];
        byte[][] normal = Pictures(Values(fixture, "normal_u8"), views, bytesEach), position = Pictures(Values(fixture, "position_u8"), views, bytesEach);
        byte[] reference = Pictures(Values(fixture, "ref_u8"), 1, bytesEach)[0];

        using CpuBackend backend = new();
        using HunyuanPaintLoader loader = HunyuanPaintLoader.Load(backend, folder, null, null, new FixedConditioner(cond, uncond, tokens, width));
        HunyuanPaintOptions options = new() { ViewResolution = size, RenderResolution = size, Steps = 1, GuidanceScale = 3f, InitialLatents = initial };
        System.Diagnostics.Stopwatch watch = System.Diagnostics.Stopwatch.StartNew();
        byte[][] generated = loader.Pipeline.Diffuse(PaintViewCandidate.Fixed, normal, position, reference, size, size, options, null, default);
        output.WriteLine($"one step on the CPU: {watch.Elapsed.TotalSeconds:F1} s");

        float[] expected = Values(fixture, "decoded_u8");
        Assert.Equal(materials * views, generated.Length);
        double sum = 0, squares = 0, reference2 = 0;
        int differing = 0, worst = 0;
        long count = 0;
        for (int image = 0; image < generated.Length; image++)
        {
            for (int i = 0; i < bytesEach; i++)
            {
                double e = expected[image * bytesEach + i], a = generated[image][i], d = a - e;
                sum += Math.Abs(d);
                squares += d * d;
                reference2 += e * e;
                differing += d != 0 ? 1 : 0;
                worst = Math.Max(worst, (int)Math.Abs(d));
                count++;
            }
        }

        double meanAbs = sum / count, relative = Math.Sqrt(squares / reference2);
        output.WriteLine($"decoded pictures against upstream (8-bit): mean |difference| {meanAbs:F3} levels, largest {worst}, {100.0 * differing / count:F2} % of the values differ, relRMS {relative:E2}");
        Assert.True(meanAbs < 1.0, $"mean difference {meanAbs:F3} levels");
        Assert.True(relative < 5e-3, $"relRMS {relative:E2}");
    }
}
