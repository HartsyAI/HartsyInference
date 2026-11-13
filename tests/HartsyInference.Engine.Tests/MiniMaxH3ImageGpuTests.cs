using System.Runtime.InteropServices;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Tests.Services;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>Device facts for the MiniMax-H3-Image path on the real int8 ConvRot build, each compared with ComfyUI 0.45.0
/// dumps. These are component checks (encoder, one denoise step, decode), not generations: a full image comes only through
/// the live SwarmUI API. They run only where the models root and the dump directory exist and CUDA is usable, and they are
/// skipped (not passed) otherwise.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("MiniMaxH3Gpu")]
public sealed class MiniMaxH3ImageGpuTests
{
    private const string ModelsDirVariable = "HARTSYINFERENCE_MODELS_DIR";
    private const string ReferenceDirVariable = "HARTSY_H3_REF_DIR";
    private const string DitRelative = "diffusion_models/MiniMaxH3Image/minimax_h3_image_pruned_int8_convrot.safetensors";
    private const string VaeRelative = "VAE/MiniMaxH3/minimax_h3_image_vae_fp16.safetensors";
    private const string TextEncoderRelative = "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors";
    private const string Prompt = "a red fox in snow";

    // Tolerances are set from the first measured run, and reported with it. Relative L2 is ||a-b|| / ||b||.
    private const double VaeRelativeL2Floor = 2e-2;
    private const double TextRelativeL2Floor = 2e-2;
    // Exact-precision reference: our f32 body against ComfyUI in fp32 with exact attention. Set from the measured run.
    // Measured 5.9e-2 with affine scale 1.00035 and offset 0.0045: quantization rounding in ~200 int8 linears compounds through
    // 50 blocks. The reference's own variants span 4-10% at this step (fp32 vs bf16, int8 matmuls kept vs not), so 0.1 is the
    // bound for this check, and it is not a claim of bit-equality.
    private const double ExactVelocityRelativeL2Floor = 1e-1;
    // Default reference: ComfyUI's bf16 body and comfy_kitchen_int8 attention. Its attention scheme alone moves the velocity by
    // about 6% (relative L2 0.060), so the bound is set above that, with the reason stated here.
    private const double DefaultVelocityRelativeL2Floor = 1e-1;
    private const double TextRefinerRelativeL2Floor = 2e-2;
    // One int8 ConvRot linear against ComfyUI's module output (bf16). Measured 8.7e-3, with affine scale 0.99996 and offset
    // ~0. ComfyUI's own module is 1.0e-2 from its float reference and 4.1e-3 from its eager formula, so this is the order of
    // activation-quantization rounding, not a structural error. Floor 2e-2.
    private const double Int8LinearRelativeL2Floor = 2e-2;

    [MiniMaxH3ImageGpuFact]
    public void VaeDecodeOfTheReferenceLatentMatchesComfyUiPixels()
    {
        string refDir = ReferenceDir();
        using CudaBackend cuda = new(0, Ernie5ModelGpuParityTests.PtxDirectory());
        List<IDisposable> loaders = [];
        try
        {
            MiniMaxH3VideoVaeDecoder decoder = MiniMaxH3ImageRecipe.LoadImageVae(Path.Combine(ModelsDir(), VaeRelative), loaders);
            using Tensor latent = new(new TensorShape([1L, 24L, 1L, 8L, 8L]), DType.F32);
            ReadInto(Path.Combine(refDir, "vae_rand_latent_f32.bin"), latent.AsSpan<float>());
            float[] expected = ReadFloats(Path.Combine(refDir, "vae_rand_pixels_f32.bin"));

            using Tensor pixels = decoder.Decode(cuda, latent);

            Assert.Equal(expected.Length, pixels.ElementCount);
            // The decoder emits [-1, 1]; ComfyUI emits [0, 1] (its de-normalised, clamped pixels). Compare in the engine's
            // convention, then check the bytes the engine would write against ComfyUI's uint8 (truncated 255 * x).
            float[] actual = pixels.AsSpan<float>().ToArray();
            float[] expectedSigned = expected.Select(v => 2f * v - 1f).ToArray();
            double relative = RelativeL2(actual, expectedSigned);
            Report("VAE decode", relative, VaeRelativeL2Floor);
            Assert.True(relative < VaeRelativeL2Floor, Diagnose("VAE decode", actual, expectedSigned, relative));

            byte[] ours = ImagePostProcessor.TensorToRgbBytes(pixels.Reshape(new TensorShape(1, 3, 128, 128)));
            // ComfyUI decodes in fp16 and we decode in f32, so the bytes may differ by up to two levels at a boundary. The
            // mean difference is the stronger check.
            int worst = 0;
            long total = 0;
            for (int y = 0; y < 128; y++)
            {
                for (int x = 0; x < 128; x++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        float reference = Math.Clamp(expected[c * 128 * 128 + y * 128 + x], 0f, 1f);
                        int comfy = (int)Math.Floor(255f * reference);
                        int difference = Math.Abs(ours[(y * 128 + x) * 3 + c] - comfy);
                        worst = Math.Max(worst, difference);
                        total += difference;
                    }
                }
            }
            double meanDifference = total / (128.0 * 128.0 * 3.0);
            Assert.True(worst <= 2 && meanDifference < 0.5,
                $"byte difference from ComfyUI's uint8 image: worst {worst}, mean {meanDifference:F3}");
        }
        finally
        {
            foreach (IDisposable loader in loaders)
            {
                loader.Dispose();
            }
        }
    }

    [MiniMaxH3ImageGpuFact]
    public void TextEncoderHiddenStatesMatchComfyUi()
    {
        string refDir = ReferenceDir();
        JsonElement manifest = ReadManifest(refDir);
        int[] expectedIds = ReadTextIds(refDir);
        float[] expectedHidden = ReadFloats(Path.Combine(refDir, "text_hidden_f32.bin"));
        int tokens = expectedIds.Length;

        using CudaBackend cuda = new(0, Ernie5ModelGpuParityTests.PtxDirectory());
        List<IDisposable> loaders = [];
        try
        {
            using MiniMaxH3TextEncoder encoder = MiniMaxH3Components.LoadTextEncoder(Path.Combine(ModelsDir(), TextEncoderRelative), loaders);
            MiniMaxH3TextEncoding.Encoded presented = MiniMaxH3ImageRecipePipeline.PresentPrompt(
                new HartsyInference.ModelAssets.Tokenizers.Qwen2Tokenizer(), Prompt);

            Assert.Equal(expectedIds, presented.TokenIds);
            System.Diagnostics.Stopwatch clock = System.Diagnostics.Stopwatch.StartNew();
            MiniMaxH3TextEncoder.Result first = encoder.Encode(cuda, presented, null);
            cuda.Sync();
            double coldMs = clock.Elapsed.TotalMilliseconds;
            first.HiddenStates.Dispose();
            clock.Restart();
            MiniMaxH3TextEncoder.Result encoded = encoder.Encode(cuda, presented, null);
            cuda.Sync();
            Console.WriteLine($"[h3-device] Text encoder encode ms: cold {coldMs:F1}, warm {clock.Elapsed.TotalMilliseconds:F1}");
            using (encoded.HiddenStates)
            {
                Assert.Equal(expectedHidden.Length, encoded.HiddenStates.ElementCount);
                double relative = RelativeL2(encoded.HiddenStates.AsSpan<float>().ToArray(), expectedHidden);
                Report("Text encoder hidden states", relative, TextRelativeL2Floor);
                Assert.True(relative < TextRelativeL2Floor,
                    $"Text-encoder relative L2 vs ComfyUI was {relative:E3} over {tokens} tokens (manifest tokens {manifest})");
            }
        }
        finally
        {
            foreach (IDisposable loader in loaders)
            {
                loader.Dispose();
            }
        }
    }

    [MiniMaxH3ImageGpuFact]
    public void TextRefinerMatchesComfyUiPreprocessedContext()
    {
        // Localises a body-side gap: ComfyUI's context is condition_proj plus the two-block token refiner, from the bf16 cast of
        // the encoder output. If this matches, the gap is in the image tokens or the body, not the text side.
        string refDir = ReferenceDir();
        int[] ids = ReadTextIds(refDir);
        float[] hidden = ReadFloats(Path.Combine(refDir, "text_hidden_f32.bin"));
        float[] refined = ReadFloats(Path.Combine(refDir, "text_refined_f32.bin"));

        using CudaBackend cuda = new(0, Ernie5ModelGpuParityTests.PtxDirectory());
        MiniMaxH3ImageRecipePipeline pipeline = (MiniMaxH3ImageRecipePipeline)new MiniMaxH3ImageRecipe().Construct(new RecipeContext
        {
            CheckpointPath = Path.Combine(ModelsDir(), DitRelative),
            Backend = cuda,
        });
        try
        {
            using Tensor textStates = new(new TensorShape([(long)ids.Length, 5120L]), DType.F32);
            CopyFloats(hidden, textStates.AsSpan<float>());
            using Tensor ours = pipeline.RefinedText(cuda, textStates);
            float[] actual = ours.AsSpan<float>().ToArray();

            Assert.Equal(refined.Length, actual.Length);
            double relative = RelativeL2(actual, refined);
            Report("Text refiner", relative, TextRefinerRelativeL2Floor);
            Assert.True(relative < TextRefinerRelativeL2Floor, Diagnose("Text refiner", actual, refined, relative));
        }
        finally
        {
            pipeline.Dispose();
        }
    }

    /// <summary>One int8 ConvRot linear on the same input as ComfyUI's module (block 0 attn.qkv_proj). It isolates the
    /// quantized matmul from the rest of the body: rotate, per-row quantize, int8 GEMM, rescale.</summary>
    [MiniMaxH3ImageGpuFact]
    public void Int8QkvLinearMatchesComfyUiOnTheSameInput()
    {
        string linearDir = Path.Combine(Path.GetDirectoryName(ReferenceDir().TrimEnd(Path.DirectorySeparatorChar))!, "h3_linear_qkv");
        float[] input = ReadFloats(Path.Combine(linearDir, "lin_in_f32.bin"));
        float[] expected = ReadFloats(Path.Combine(linearDir, "lin_out_f32.bin"));

        using CudaBackend cuda = new(0, Ernie5ModelGpuParityTests.PtxDirectory());
        List<IDisposable> loaders = [];
        try
        {
            using MiniMaxH3Transformer transformer = MiniMaxH3Components.LoadTransformer(
                Path.Combine(ModelsDir(), DitRelative), loaders, DType.F32, [cuda], out _, out Dictionary<string, Tensor> weights);
            Tensor qkv = weights["blocks.0.attn.qkv_proj.weight"];
            Assert.Equal(DType.I8, qkv.DType);

            using Tensor x = new(new TensorShape([64L, 5376L]), DType.F32);
            CopyFloats(input, x.AsSpan<float>());
            using Tensor y = new(new TensorShape([64L, 21504L]), DType.F32);
            cuda.Linear(y, x, qkv, null);

            float[] actual = y.AsSpan<float>().ToArray();
            double relative = RelativeL2(actual, expected);
            Report("int8 qkv linear", relative, Int8LinearRelativeL2Floor);
            Assert.True(relative < Int8LinearRelativeL2Floor, Diagnose("int8 qkv linear", actual, expected, relative));
        }
        finally
        {
            foreach (IDisposable loader in loaders)
            {
                loader.Dispose();
            }
        }
    }

    /// <summary>The structural check: our f32 body with exact attention against ComfyUI run in fp32 with exact SDPA. It is the
    /// only reference that removes the reference's own bf16 body and int8 attention, so it is the floor for the body itself.</summary>
    [MiniMaxH3ImageExactReferenceFact]
    public void DenoiseStepVelocityMatchesExactComfyUiAtStepZero()
    {
        AssertDenoiseStepMatches(step: 0, inputsDir: ReferenceDir(), targetDir: ExactReferenceDir(), floor: ExactVelocityRelativeL2Floor);
    }

    /// <summary>The bounded check: against the default ComfyUI dump (bf16 body, comfy_kitchen_int8 attention). The gap is the
    /// reference's own approximation, measured as 6% from its attention scheme alone, so the floor is set to that reason.</summary>
    [MiniMaxH3ImageGpuFact]
    public void DenoiseStepVelocityMatchesDefaultComfyUiAtStepZeroWithinItsOwnPrecision()
    {
        AssertDenoiseStepMatches(step: 0, inputsDir: ReferenceDir(), targetDir: ReferenceDir(), floor: DefaultVelocityRelativeL2Floor);
    }

    [MiniMaxH3ImageGpuFact]
    public void DenoiseStepVelocityMatchesComfyUiAtTheLastStep()
    {
        AssertDenoiseStepMatches(step: 5, inputsDir: ReferenceDir(), targetDir: ReferenceDir(), floor: DefaultVelocityRelativeL2Floor);
    }

    private static void AssertDenoiseStepMatches(int step, string inputsDir, string targetDir, double floor)
    {
        string refDir = inputsDir;
        JsonElement manifest = ReadManifest(refDir);
        double[] sigmas = manifest.GetProperty("sigmas_7_float64").EnumerateArray().Select(e => e.GetDouble()).ToArray();
        int[] expectedIds = ReadTextIds(refDir);
        int tokens = expectedIds.Length;
        float[] hidden = ReadFloats(Path.Combine(refDir, "text_hidden_f32.bin"));
        float[] xValues = ReadFloats(Path.Combine(refDir, $"step{step}_x_f32.bin"));
        float[] velocityValues = ReadFloats(Path.Combine(targetDir, $"step{step}_model_out_f32.bin"));

        using CudaBackend cuda = new(0, Ernie5ModelGpuParityTests.PtxDirectory());
        RecipeContext context = new()
        {
            CheckpointPath = Path.Combine(ModelsDir(), DitRelative),
            Backend = cuda,
        };
        MiniMaxH3ImageRecipePipeline pipeline = (MiniMaxH3ImageRecipePipeline)new MiniMaxH3ImageRecipe().Construct(context);
        try
        {
            MiniMaxH3PackedLayout layout = new(tokens, 1, 16, 16, 0);
            using Tensor latent = new(new TensorShape([1L, 24L, 1L, 16L, 16L]), DType.F32);
            CopyFloats(xValues, latent.AsSpan<float>());
            using Tensor x = MiniMaxH3Latents.PackVideo(latent, pipeline.Config);
            using Tensor referenceLatent = new(new TensorShape([1L, 24L, 1L, 16L, 16L]), DType.F32);
            CopyFloats(velocityValues, referenceLatent.AsSpan<float>());
            using Tensor referenceVelocity = MiniMaxH3Latents.PackVideo(referenceLatent, pipeline.Config);
            using Tensor textStates = new(new TensorShape([(long)tokens, 5120L]), DType.F32);
            CopyFloats(hidden, textStates.AsSpan<float>());
            MiniMaxH3TextEncoder.Result encoded = new()
            {
                HiddenStates = textStates,
                TagRuns = [(0, tokens, MiniMaxH3TextEncoding.TextTag)],
            };
            // The rope tables are owned by the caller; the stacked head belongs to the transformer and is not disposed here.
            (Tensor cos, Tensor sin) = pipeline.RopeTables(layout);
            (Tensor weight, Tensor bias) = pipeline.StackedHead();
            using (cos)
            using (sin)
            {
                float[] xBefore = x.AsSpan<float>().ToArray();
                Tensor input = CopyOf(x);
                Tensor next = pipeline.DenoiseStep(input, layout, encoded, cos, sin, weight, bias,
                    sigmas[step], sigmas[step + 1]);
                using (next)
                {
                    // The sampler's Euler update is next = x + dt * v, so the velocity the pipeline used is (next - x) / dt.
                    float dt = (float)(sigmas[step + 1] - sigmas[step]);
                    float[] nextValues = next.AsSpan<float>().ToArray();
                    float[] velocity = new float[nextValues.Length];
                    for (int i = 0; i < velocity.Length; i++)
                    {
                        velocity[i] = (nextValues[i] - xBefore[i]) / dt;
                    }
                    float[] referenceVelocityValues = referenceVelocity.AsSpan<float>().ToArray();
                    double relative = RelativeL2(velocity, referenceVelocityValues);
                    Report($"DiT step {step} velocity vs {Path.GetFileName(targetDir)}", relative, floor);
                    Assert.True(relative < floor,
                        Diagnose($"DiT step {step} velocity", velocity, referenceVelocityValues, relative));
                }
            }
        }
        finally
        {
            pipeline.Dispose();
        }
    }

    // ---- Helpers ------------------------------------------------------------------------------------------------

    private static Tensor CopyOf(Tensor source)
    {
        Tensor copy = new(source.Shape, DType.F32);
        source.AsSpan<float>().CopyTo(copy.AsSpan<float>());
        return copy;
    }

    private static string ReferenceDir()
    {
        string? dir = Environment.GetEnvironmentVariable(ReferenceDirVariable);
        return dir ?? throw new InvalidOperationException($"{ReferenceDirVariable} is not set.");
    }

    private static string ExactReferenceDir()
    {
        string root = ReferenceDir();
        return Path.Combine(Path.GetDirectoryName(root.TrimEnd(Path.DirectorySeparatorChar))!, Path.GetFileName(root) + "_exact_fp32");
    }

    private static string ModelsDir() =>
        Environment.GetEnvironmentVariable(ModelsDirVariable)
        ?? throw new InvalidOperationException($"{ModelsDirVariable} is not set.");

    /// <summary>The prompt's token ids as ComfyUI's node presented them: the first (only) entry of <c>node_ids</c>.</summary>
    private static int[] ReadTextIds(string refDir)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(refDir, "text_ids.json")));
        return document.RootElement.GetProperty("node_ids")[0].EnumerateArray().Select(e => e.GetInt32()).ToArray();
    }

    private static JsonElement ReadManifest(string refDir)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(refDir, "manifest.json")));
        return document.RootElement.Clone();
    }

    private static float[] ReadFloats(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
    }

    private static void ReadInto(string path, Span<float> destination) => CopyFloats(ReadFloats(path), destination);

    private static void CopyFloats(float[] source, Span<float> destination)
    {
        if (source.Length != destination.Length)
        {
            throw new InvalidDataException($"Expected {destination.Length} values, found {source.Length}.");
        }
        source.AsSpan().CopyTo(destination);
    }

    /// <summary>Prints the measured value so a passing run still records it in the test output.</summary>
    private static void Report(string what, double relative, double floor) =>
        Console.WriteLine($"[h3-device] {what}: relative L2 {relative:E3} (floor {floor:E1})");

    /// <summary>A failure message that says whether the gap is a convention (an affine map fits) or a structural error.</summary>
    private static string Diagnose(string what, float[] actual, float[] expected, double relative)
    {
        double sa = 0, se = 0, saa = 0, sae = 0, n = expected.Length;
        double minA = double.MaxValue, maxA = double.MinValue, minE = double.MaxValue, maxE = double.MinValue;
        for (int i = 0; i < expected.Length; i++)
        {
            double a = actual[i], e = expected[i];
            sa += a; se += e; saa += a * a; sae += a * e;
            minA = Math.Min(minA, a); maxA = Math.Max(maxA, a);
            minE = Math.Min(minE, e); maxE = Math.Max(maxE, e);
        }
        double denominator = n * saa - sa * sa;
        double scale = denominator == 0 ? 0 : (n * sae - sa * se) / denominator;
        double offset = (se - scale * sa) / n;
        double fitted = 0, norm = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = scale * actual[i] + offset - expected[i];
            fitted += d * d;
            norm += (double)expected[i] * expected[i];
        }
        return $"{what} relative L2 vs ComfyUI was {relative:E3}. Ours range [{minA:F4}, {maxA:F4}], ComfyUI range "
            + $"[{minE:F4}, {maxE:F4}]. Best affine fit ours->ComfyUI: scale {scale:F5}, offset {offset:F5}, residual "
            + $"relative L2 {Math.Sqrt(fitted) / Math.Max(1e-30, Math.Sqrt(norm)):E3}.";
    }

    private static double RelativeL2(float[] actual, float[] expected)
    {
        double diff = 0, norm = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - expected[i];
            diff += d * d;
            norm += (double)expected[i] * expected[i];
        }
        return Math.Sqrt(diff) / Math.Max(1e-30, Math.Sqrt(norm));
    }
}

/// <summary>A GPU fact that also needs the exact-precision ComfyUI dump beside the default one (<c>&lt;dir&gt;_exact_fp32</c>).</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class MiniMaxH3ImageExactReferenceFactAttribute : FactAttribute
{
    public MiniMaxH3ImageExactReferenceFactAttribute()
    {
        string? refs = Environment.GetEnvironmentVariable("HARTSY_H3_REF_DIR");
        if (string.IsNullOrWhiteSpace(refs) || !File.Exists(Path.Combine(refs.TrimEnd(Path.DirectorySeparatorChar) + "_exact_fp32", "step0_model_out_f32.bin")))
        {
            Skip = "SKIPPED: the exact-precision ComfyUI dump (<HARTSY_H3_REF_DIR>_exact_fp32) is not present; this structural check did not run.";
        }
    }
}

/// <summary>A GPU fact that runs only with <c>HARTSYINFERENCE_MODELS_DIR</c> holding the int8 ConvRot DiT, the image VAE
/// and the text encoder, <c>HARTSY_H3_REF_DIR</c> holding the ComfyUI dumps' manifest, and a usable CUDA device.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class MiniMaxH3ImageGpuFactAttribute : FactAttribute
{
    public MiniMaxH3ImageGpuFactAttribute()
    {
        string? models = Environment.GetEnvironmentVariable("HARTSYINFERENCE_MODELS_DIR");
        string? refs = Environment.GetEnvironmentVariable("HARTSY_H3_REF_DIR");
        if (string.IsNullOrWhiteSpace(models)
            || !File.Exists(Path.Combine(models, "diffusion_models", "MiniMaxH3Image", "minimax_h3_image_pruned_int8_convrot.safetensors")))
        {
            Skip = "SKIPPED: HARTSYINFERENCE_MODELS_DIR does not hold the int8 ConvRot image DiT; the device check did not run.";
        }
        else if (string.IsNullOrWhiteSpace(refs) || !File.Exists(Path.Combine(refs, "manifest.json")))
        {
            Skip = "SKIPPED: HARTSY_H3_REF_DIR has no manifest.json (ComfyUI dumps); the device check did not run.";
        }
        else if (!CudaContext.IsAvailable())
        {
            Skip = "SKIPPED: no usable CUDA device; the device check did not run.";
        }
    }
}
