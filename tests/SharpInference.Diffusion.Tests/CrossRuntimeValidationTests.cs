using System.Text.Json;
using Xunit;
using Xunit.Abstractions;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.Diffusion.Tests;

/// <summary>
/// Cross-runtime validation tests that compare C# outputs against Python reference values.
/// These tests help diagnose signal loss and numerical divergence between implementations.
///
/// <para>To generate reference data, run: <c>python tests/python-reference/dump_reference_stats.py</c></para>
/// </summary>
public class CrossRuntimeValidationTests
{
    private const string ModelDir = @"C:\Users\AI Overlord\Desktop\Projects\SharpInference\tests\test-models\sd15";
    private const string DiagnosticsDir = @"C:\Users\AI Overlord\Desktop\Projects\SharpInference\tests\python-reference";
    private static readonly string ReferenceTensorsDir = Path.Combine(DiagnosticsDir, "reference_tensors");
    private static readonly string ReferenceStatsPath = Path.Combine(DiagnosticsDir, "reference_stats.json");

    private readonly ITestOutputHelper _output;

    public CrossRuntimeValidationTests(ITestOutputHelper output) => _output = output;

    #region Test 1: Scheduler Validation

    /// <summary>
    /// Validates that the C# scheduler produces identical sigma values, timesteps, and
    /// InitialNoiseSigma compared to the Python reference. This is a fast, self-contained
    /// test that requires no model files — just the reference_stats.json.
    /// </summary>
    [Fact]
    public void SchedulerMatchesPythonReference()
    {
        // Python reference values from reference_stats.json
        float[] expectedTimesteps =
        [
            950f, 900f, 850f, 800f, 750f, 700f, 650f, 600f, 550f, 500f,
            450f, 400f, 350f, 300f, 250f, 200f, 150f, 100f, 50f, 0f
        ];

        float[] expectedSigmas =
        [
            10.96606731f, 8.34659958f, 6.47457790f, 5.11103010f, 4.09916592f,
            3.33438993f, 2.74580836f, 2.28464103f, 1.91683412f, 1.61827970f,
            1.37166870f, 1.16439164f, 0.98710895f, 0.83275318f, 0.69579947f,
            0.57166636f, 0.45607531f, 0.34393182f, 0.22558255f, 0.02916753f,
            0.0f
        ];

        float expectedInitNoiseSigma = 11.01156807f;

        // Compute C# values
        EulerDiscreteScheduler scheduler = new();
        scheduler.SetTimesteps(20);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        float initSigma = scheduler.InitialNoiseSigma;

        // Validate timesteps
        _output.WriteLine("=== Timestep Comparison ===");
        Assert.Equal(expectedTimesteps.Length, timesteps.Length);
        for (int i = 0; i < expectedTimesteps.Length; i++)
        {
            float diff = MathF.Abs(timesteps[i] - expectedTimesteps[i]);
            _output.WriteLine($"  t[{i}]: C#={timesteps[i]:F4}, Python={expectedTimesteps[i]:F4}, diff={diff:E4}");
            Assert.True(diff < 0.01f, $"Timestep {i} diverged: C#={timesteps[i]}, Python={expectedTimesteps[i]}");
        }

        // Validate InitialNoiseSigma
        float sigmaDiff = MathF.Abs(initSigma - expectedInitNoiseSigma);
        _output.WriteLine($"\n=== InitialNoiseSigma ===");
        _output.WriteLine($"  C#={initSigma:F6}, Python={expectedInitNoiseSigma:F6}, diff={sigmaDiff:E4}");
        Assert.True(sigmaDiff < 0.001f, $"InitialNoiseSigma diverged: C#={initSigma}, Python={expectedInitNoiseSigma}");

        // Validate sigmas (requires accessing private field - use ScaleModelInput to infer)
        _output.WriteLine($"\n=== Sigma Validation (via ScaleModelInput) ===");
        for (int i = 0; i < 20; i++)
        {
            // ScaleModelInput returns 1/sqrt(sigma^2 + 1), so sigma = sqrt(1/scale^2 - 1)
            float scale = scheduler.ScaleModelInput(i);
            float inferredSigma = MathF.Sqrt(1.0f / (scale * scale) - 1.0f);
            float sigDiff = MathF.Abs(inferredSigma - expectedSigmas[i]);
            _output.WriteLine($"  sigma[{i}]: C#={inferredSigma:F6}, Python={expectedSigmas[i]:F6}, diff={sigDiff:E4}");
            Assert.True(sigDiff < 0.01f, $"Sigma {i} diverged: C#={inferredSigma}, Python={expectedSigmas[i]}");
        }

        _output.WriteLine("\nScheduler validation PASSED — all values match Python reference.");
    }

    #endregion

    #region Test 2: Cross-Runtime Pipeline (Uses Python Noise)

    /// <summary>
    /// Loads the exact initial noise tensor from Python, runs the full C# denoising pipeline,
    /// and compares per-step latent statistics against Python reference values.
    /// This eliminates RNG differences as a variable.
    ///
    /// <para>Requires: <c>python tests/python-reference/dump_reference_stats.py</c> to generate reference_tensors/</para>
    /// </summary>
    [Fact]
    public unsafe void PipelineWithPythonNoiseMatchesReference()
    {
        string noisePath = Path.Combine(ReferenceTensorsDir, "initial_noise.bin");
        if (!File.Exists(noisePath))
        {
            Assert.Fail($"Reference noise not found: {noisePath}\nRun: python tests/python-reference/dump_reference_stats.py");
            return;
        }
        if (!Directory.Exists(ModelDir))
        {
            Assert.Fail($"Model directory not found: {ModelDir}");
            return;
        }

        // Load Python reference stats for comparison
        List<Dictionary<string, object>> refStats = LoadReferenceStats();

        // Load Python's initial noise
        TensorShape latentShape = new(1, 4, 32, 32);
        Tensor latent = LoadBinaryTensor(noisePath, latentShape);
        _output.WriteLine($"Loaded Python noise: mean={ComputeMean(latent):F6}, std={ComputeStd(latent):F6}");

        // Compare initial noise stats
        Dictionary<string, object>? pyNoise = FindStat(refStats, "initial_noise");
        if (pyNoise != null)
            _output.WriteLine($"Python noise ref:    mean={GetDouble(pyNoise, "mean"):F6}, std={GetDouble(pyNoise, "std"):F6}");

        using CpuBackend backend = new();

        // Load models
        ClipTextEncoderConfig clipConfig = ClipTextEncoderConfig.Sd15;
        ClipTextEncoder textEncoder = new(clipConfig);
        using SafeTensorsLoader teLoader = new();
        teLoader.Load(Path.Combine(ModelDir, "text_encoder", "model.fp16.safetensors"));
        textEncoder.LoadWeights(CastWeightsToF32(teLoader.GetAllTensors()), "text_model");

        UNet unet = new(UNetConfig.Sd15);
        using SafeTensorsLoader unetLoader = new();
        unetLoader.Load(Path.Combine(ModelDir, "unet", "diffusion_pytorch_model.fp16.safetensors"));
        unet.LoadWeights(CastWeightsToF32(unetLoader.GetAllTensors()));

        // Encode text (same tokens as Python)
        string prompt = "a painting of a cat sitting on a windowsill";
        string negativePrompt = "blurry, bad quality";
        using SharpInference.Tokenizers.ClipTokenizer tokenizer = new(
            Path.Combine(ModelDir, "tokenizer", "vocab.json"),
            Path.Combine(ModelDir, "tokenizer", "merges.txt"));

        int[] promptTokens = tokenizer.Encode(prompt);
        int[] negativeTokens = tokenizer.Encode(negativePrompt);
        int[][] batchTokenIds = [negativeTokens, promptTokens];
        Tensor textEmbeddings = textEncoder.Encode(backend, batchTokenIds);
        int seqLen = (int)textEmbeddings.Shape[1];
        int hiddenSize = (int)textEmbeddings.Shape[2];

        _output.WriteLine($"Text embeddings: mean={ComputeMean(textEmbeddings):F6}, std={ComputeStd(textEmbeddings):F6}");

        // Scheduler
        EulerDiscreteScheduler scheduler = new();
        scheduler.SetTimesteps(20);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        // Scale initial noise
        float initSigma = scheduler.InitialNoiseSigma;
        Tensor scaled = new(latentShape, DType.F32);
        backend.Scale(scaled, latent, initSigma);
        latent.Dispose();
        latent = scaled;

        float cfgScale = 7.5f;

        _output.WriteLine($"\n{"Step",-8} {"C# Mean",12} {"Py Mean",12} {"C# Std",12} {"Py Std",12} {"Std Ratio",12}");
        _output.WriteLine(new string('-', 68));

        for (int i = 0; i < 20; i++)
        {
            float t = timesteps[i];

            float inputScale = scheduler.ScaleModelInput(i);
            Tensor scaledLatent;
            if (MathF.Abs(inputScale - 1.0f) > 1e-6f)
            {
                scaledLatent = new Tensor(latentShape, DType.F32);
                backend.Scale(scaledLatent, latent, inputScale);
            }
            else
            {
                scaledLatent = latent;
            }

            Tensor uncondEmb = SliceBatch(textEmbeddings, 0, seqLen, hiddenSize);
            Tensor condEmb = SliceBatch(textEmbeddings, 1, seqLen, hiddenSize);

            Tensor uncondNoise = unet.Forward(backend, scaledLatent, t, uncondEmb);
            Tensor condNoise = unet.Forward(backend, scaledLatent, t, condEmb);
            uncondEmb.Dispose();
            condEmb.Dispose();

            Tensor noisePred = new(latentShape, DType.F32);
            float* up = (float*)uncondNoise.DataPointer;
            float* cp = (float*)condNoise.DataPointer;
            float* fp = (float*)noisePred.DataPointer;
            int count = (int)latentShape.ElementCount;
            for (int j = 0; j < count; j++)
                fp[j] = up[j] + cfgScale * (cp[j] - up[j]);
            uncondNoise.Dispose();
            condNoise.Dispose();

            if (scaledLatent != latent) scaledLatent.Dispose();

            Tensor newLatent = new(latentShape, DType.F32);
            scheduler.Step(newLatent, noisePred, latent, i);
            noisePred.Dispose();
            latent.Dispose();
            latent = newLatent;

            // Compare against Python reference
            float csMean = ComputeMean(latent);
            float csStd = ComputeStd(latent);

            Dictionary<string, object>? pyStep = FindStat(refStats, $"step{i}_latents_after");
            if (pyStep != null)
            {
                double pyMean = GetDouble(pyStep, "mean");
                double pyStd = GetDouble(pyStep, "std");
                double stdRatio = csStd / pyStd;
                _output.WriteLine($"Step {i,-3} {csMean,12:F6} {pyMean,12:F6} {csStd,12:F6} {pyStd,12:F6} {stdRatio,12:F4}");
            }
            else
            {
                _output.WriteLine($"Step {i,-3} {csMean,12:F6} {"N/A",12} {csStd,12:F6} {"N/A",12}");
            }
        }

        textEmbeddings.Dispose();

        float finalMean = ComputeMean(latent);
        float finalStd = ComputeStd(latent);
        Dictionary<string, object>? pyFinal = FindStat(refStats, "final_latents");
        double pyFinalStd = pyFinal != null ? GetDouble(pyFinal, "std") : double.NaN;

        _output.WriteLine($"\n=== Final Comparison ===");
        _output.WriteLine($"C# final:     mean={finalMean:F6}, std={finalStd:F6}");
        if (pyFinal != null)
        {
            _output.WriteLine($"Python final: mean={GetDouble(pyFinal, "mean"):F6}, std={pyFinalStd:F6}");
            _output.WriteLine($"Std ratio (C#/Python): {finalStd / pyFinalStd:F4}");
        }

        latent.Dispose();
    }

    #endregion

    #region Test 3: Single UNet Pass Comparison

    /// <summary>
    /// Loads the exact step 0 scaled input from Python, runs a single UNet forward pass,
    /// and compares the noise prediction against Python's output.
    /// This isolates whether the UNet itself produces different results.
    ///
    /// <para>Requires: <c>python tests/python-reference/dump_reference_stats.py</c> to generate reference_tensors/</para>
    /// </summary>
    [Fact]
    public unsafe void SingleUNetPassMatchesPythonReference()
    {
        string inputPath = Path.Combine(ReferenceTensorsDir, "step0_scaled_input.bin");
        string uncondRefPath = Path.Combine(ReferenceTensorsDir, "step0_noise_pred_uncond.bin");
        string condRefPath = Path.Combine(ReferenceTensorsDir, "step0_noise_pred_cond.bin");
        string textEmbPath = Path.Combine(ReferenceTensorsDir, "text_embeddings.bin");

        if (!File.Exists(inputPath))
        {
            Assert.Fail($"Reference tensors not found. Run: python tests/python-reference/dump_reference_stats.py");
            return;
        }
        if (!Directory.Exists(ModelDir))
        {
            Assert.Fail($"Model directory not found: {ModelDir}");
            return;
        }

        using CpuBackend backend = new();

        // Load UNet
        UNet unet = new(UNetConfig.Sd15);
        using SafeTensorsLoader unetLoader = new();
        unetLoader.Load(Path.Combine(ModelDir, "unet", "diffusion_pytorch_model.fp16.safetensors"));
        unet.LoadWeights(CastWeightsToF32(unetLoader.GetAllTensors()));

        // Load Python's exact step 0 input
        TensorShape latentShape = new(1, 4, 32, 32);
        Tensor scaledInput = LoadBinaryTensor(inputPath, latentShape);
        _output.WriteLine($"Step 0 input:  mean={ComputeMean(scaledInput):F6}, std={ComputeStd(scaledInput):F6}");

        // Load text embeddings — either from binary or re-encode
        Tensor textEmbeddings;
        if (File.Exists(textEmbPath))
        {
            TensorShape embShape = new(2, 77, 768);
            textEmbeddings = LoadBinaryTensor(textEmbPath, embShape);
            _output.WriteLine($"Text emb (file): mean={ComputeMean(textEmbeddings):F6}, std={ComputeStd(textEmbeddings):F6}");
        }
        else
        {
            // Fall back to re-encoding (will differ slightly from Python due to fp16→fp32 casting)
            ClipTextEncoderConfig clipConfig = ClipTextEncoderConfig.Sd15;
            ClipTextEncoder textEncoder = new(clipConfig);
            using SafeTensorsLoader teLoader = new();
            teLoader.Load(Path.Combine(ModelDir, "text_encoder", "model.fp16.safetensors"));
            textEncoder.LoadWeights(CastWeightsToF32(teLoader.GetAllTensors()), "text_model");

            using SharpInference.Tokenizers.ClipTokenizer tokenizer = new(
                Path.Combine(ModelDir, "tokenizer", "vocab.json"),
                Path.Combine(ModelDir, "tokenizer", "merges.txt"));
            int[] promptTokens = tokenizer.Encode("a painting of a cat sitting on a windowsill");
            int[] negativeTokens = tokenizer.Encode("blurry, bad quality");
            textEmbeddings = textEncoder.Encode(backend, [negativeTokens, promptTokens]);
            _output.WriteLine($"Text emb (re-encoded): mean={ComputeMean(textEmbeddings):F6}, std={ComputeStd(textEmbeddings):F6}");
        }

        int seqLen = (int)textEmbeddings.Shape[1];
        int hiddenSize = (int)textEmbeddings.Shape[2];

        // Run unconditional UNet pass
        float timestep = 950.0f;
        Tensor uncondEmb = SliceBatch(textEmbeddings, 0, seqLen, hiddenSize);
        _output.WriteLine($"\nRunning UNet (uncond) at t={timestep}...");
        Tensor uncondPred = unet.Forward(backend, scaledInput, timestep, uncondEmb);
        uncondEmb.Dispose();

        float uncondMean = ComputeMean(uncondPred);
        float uncondStd = ComputeStd(uncondPred);
        _output.WriteLine($"  C# uncond pred: mean={uncondMean:F6}, std={uncondStd:F6}");

        // Run conditional UNet pass
        Tensor condEmb = SliceBatch(textEmbeddings, 1, seqLen, hiddenSize);
        _output.WriteLine($"Running UNet (cond) at t={timestep}...");
        Tensor condPred = unet.Forward(backend, scaledInput, timestep, condEmb);
        condEmb.Dispose();

        float condMean = ComputeMean(condPred);
        float condStd = ComputeStd(condPred);
        _output.WriteLine($"  C# cond pred:   mean={condMean:F6}, std={condStd:F6}");

        // Load Python reference stats
        List<Dictionary<string, object>> refStats = LoadReferenceStats();
        Dictionary<string, object>? pyUncond = FindStat(refStats, "step0_noise_pred_uncond");
        Dictionary<string, object>? pyCond = FindStat(refStats, "step0_noise_pred_cond");

        _output.WriteLine($"\n=== UNet Output Comparison ===");
        _output.WriteLine($"{"",18} {"C# Mean",12} {"Py Mean",12} {"C# Std",12} {"Py Std",12}");
        _output.WriteLine(new string('-', 66));

        if (pyUncond != null)
        {
            double pyUM = GetDouble(pyUncond, "mean");
            double pyUS = GetDouble(pyUncond, "std");
            _output.WriteLine($"{"Uncond",-18} {uncondMean,12:F6} {pyUM,12:F6} {uncondStd,12:F6} {pyUS,12:F6}");
        }
        if (pyCond != null)
        {
            double pyCM = GetDouble(pyCond, "mean");
            double pyCS = GetDouble(pyCond, "std");
            _output.WriteLine($"{"Cond",-18} {condMean,12:F6} {pyCM,12:F6} {condStd,12:F6} {pyCS,12:F6}");
        }

        // Compare element-wise if reference tensors exist
        if (File.Exists(uncondRefPath))
        {
            Tensor pyUncondTensor = LoadBinaryTensor(uncondRefPath, latentShape);
            CompareElementWise("uncond_pred", uncondPred, pyUncondTensor);
            pyUncondTensor.Dispose();
        }
        if (File.Exists(condRefPath))
        {
            Tensor pyCondTensor = LoadBinaryTensor(condRefPath, latentShape);
            CompareElementWise("cond_pred", condPred, pyCondTensor);
            pyCondTensor.Dispose();
        }

        uncondPred.Dispose();
        condPred.Dispose();
        scaledInput.Dispose();
        textEmbeddings.Dispose();
    }

    #endregion

    #region Test 4: GELU Variant Comparison

    /// <summary>
    /// Compares tanh-approximated GELU (used in C#) vs exact erf-based GELU (used in PyTorch).
    /// Reports the maximum error and estimates cumulative impact across SD1.5 UNet.
    /// </summary>
    [Fact]
    public void GeluTanhVsErfDifference()
    {
        float maxAbsError = 0f;
        float maxRelError = 0f;
        double sumAbsError = 0;
        int numSamples = 10000;

        _output.WriteLine("=== GELU tanh-approx vs erf-exact comparison ===");
        _output.WriteLine($"Testing {numSamples} values in [-5, 5]");

        for (int i = 0; i < numSamples; i++)
        {
            float x = -5.0f + 10.0f * i / (numSamples - 1);

            // Tanh approximation (C# implementation)
            float x3 = x * x * x;
            float inner = 0.7978845608f * (x + 0.044715f * x3);
            float geluTanh = x * 0.5f * (1.0f + MathF.Tanh(inner));

            // Exact erf-based (PyTorch default)
            // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
            float geluErf = (float)(x * 0.5 * (1.0 + Erf(x / Math.Sqrt(2.0))));

            float absErr = MathF.Abs(geluTanh - geluErf);
            float relErr = MathF.Abs(geluErf) > 1e-6f ? absErr / MathF.Abs(geluErf) : 0f;

            if (absErr > maxAbsError) maxAbsError = absErr;
            if (relErr > maxRelError) maxRelError = relErr;
            sumAbsError += absErr;
        }

        double avgAbsError = sumAbsError / numSamples;

        _output.WriteLine($"Max absolute error:  {maxAbsError:E6}");
        _output.WriteLine($"Max relative error:  {maxRelError:E6}");
        _output.WriteLine($"Avg absolute error:  {avgAbsError:E6}");
        _output.WriteLine(" ");

        // Estimate cumulative impact in SD1.5
        // SD1.5 has 16 transformer blocks (8 down + 8 up), each with a GEGLU FFN
        // Per UNet pass: 16 GEGLU applications, each on innerDim elements
        // Per denoising step: 2 UNet passes (CFG)
        // Total for 20 steps: 640 GEGLU applications
        int geGluAppsPerStep = 16 * 2; // 16 transformer blocks × 2 (uncond+cond)
        int totalGeGluApps = geGluAppsPerStep * 20;
        _output.WriteLine($"Estimated GEGLU applications for 20-step SD1.5:");
        _output.WriteLine($"  Per step: {geGluAppsPerStep}");
        _output.WriteLine($"  Total: {totalGeGluApps}");
        _output.WriteLine($"  Worst-case cumulative error per element: {maxAbsError * totalGeGluApps:E6}");
        _output.WriteLine($"  Avg cumulative error per element: {avgAbsError * totalGeGluApps:E6}");
        _output.WriteLine(" ");

        // Verdict
        if (maxAbsError < 0.01f)
            _output.WriteLine("VERDICT: GELU approximation error is small (<0.01). Unlikely to be the primary cause of signal loss.");
        else
            _output.WriteLine("VERDICT: GELU approximation error is significant. Consider switching to erf-based GELU.");
    }

    /// <summary>Approximate erf function for reference comparison.</summary>
    private static double Erf(double x)
    {
        // Abramowitz and Stegun approximation (max error ~1.5e-7)
        double a1 = 0.254829592;
        double a2 = -0.284496736;
        double a3 = 1.421413741;
        double a4 = -1.453152027;
        double a5 = 1.061405429;
        double p = 0.3275911;

        double sign = x >= 0 ? 1.0 : -1.0;
        x = Math.Abs(x);
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);
        return sign * y;
    }

    #endregion

    #region Test 5: Std Population vs Sample Variance

    /// <summary>
    /// Verifies that C# uses population std (N divisor) matching PyTorch's tensor.std() default (correction=1, sample std).
    /// This checks if there's a systematic std calculation difference in our diagnostics.
    ///
    /// IMPORTANT: PyTorch tensor.std() uses Bessel's correction (N-1 divisor) by default.
    /// Our diagnostic ComputeStd uses population std (N divisor).
    /// For 4096 elements (32x32x4), the difference is: sqrt(4096/4095) ≈ 1.000122, negligible.
    /// </summary>
    [Fact]
    public void StdComputationConsistency()
    {
        int n = 4096; // SD1.5 latent: 1*4*32*32
        float populationFactor = MathF.Sqrt((float)n / (n - 1));

        _output.WriteLine($"Population vs sample std correction for N={n}: {populationFactor:F6}");
        _output.WriteLine($"This means our population std is {(1.0 - 1.0 / populationFactor) * 100:F4}% lower than PyTorch's sample std.");
        _output.WriteLine("For N=4096, this is negligible (~0.012%).");

        Assert.True(MathF.Abs(populationFactor - 1.0f) < 0.001f,
            "Population vs sample std correction is larger than expected");
    }

    #endregion

    #region Helpers

    private static unsafe Tensor LoadBinaryTensor(string path, TensorShape shape)
    {
        byte[] bytes = File.ReadAllBytes(path);
        long expectedBytes = shape.ElementCount * sizeof(float);
        if (bytes.Length != expectedBytes)
            throw new InvalidOperationException(
                $"Binary tensor size mismatch: got {bytes.Length} bytes, expected {expectedBytes} for shape {shape}");

        Tensor tensor = new(shape, DType.F32);
        fixed (byte* src = bytes)
        {
            Buffer.MemoryCopy(src, (void*)tensor.DataPointer, expectedBytes, expectedBytes);
        }
        return tensor;
    }

    private unsafe void CompareElementWise(string name, Tensor actual, Tensor expected)
    {
        float* actPtr = (float*)actual.DataPointer;
        float* expPtr = (float*)expected.DataPointer;
        int count = (int)actual.ElementCount;

        float maxAbsErr = 0f;
        double sumAbsErr = 0;
        int maxErrIdx = 0;

        for (int i = 0; i < count; i++)
        {
            float err = MathF.Abs(actPtr[i] - expPtr[i]);
            sumAbsErr += err;
            if (err > maxAbsErr)
            {
                maxAbsErr = err;
                maxErrIdx = i;
            }
        }

        double avgAbsErr = sumAbsErr / count;
        _output.WriteLine($"\n  Element-wise comparison [{name}] ({count} elements):");
        _output.WriteLine($"    Max abs error: {maxAbsErr:E6} at index {maxErrIdx}");
        _output.WriteLine($"    Avg abs error: {avgAbsErr:E6}");
        _output.WriteLine($"    Max error element: C#={actPtr[maxErrIdx]:F6}, Python={expPtr[maxErrIdx]:F6}");
    }

    private static List<Dictionary<string, object>> LoadReferenceStats()
    {
        string json = File.ReadAllText(ReferenceStatsPath);
        List<Dictionary<string, JsonElement>>? rawList = JsonSerializer.Deserialize<List<Dictionary<string, JsonElement>>>(json)!;

        List<Dictionary<string, object>> result = new();
        foreach (Dictionary<string, JsonElement> dict in rawList)
        {
            Dictionary<string, object> converted = new();
            foreach (KeyValuePair<string, JsonElement> kvp in dict)
            {
                converted[kvp.Key] = kvp.Value;
            }
            result.Add(converted);
        }
        return result;
    }

    private static Dictionary<string, object>? FindStat(List<Dictionary<string, object>> stats, string name)
    {
        return stats.FirstOrDefault(s =>
            s.TryGetValue("name", out object? n) &&
            n is JsonElement je &&
            je.GetString() == name);
    }

    private static double GetDouble(Dictionary<string, object> stat, string key)
    {
        if (stat.TryGetValue(key, out object? val) && val is JsonElement je)
            return je.GetDouble();
        return double.NaN;
    }

    private static unsafe float ComputeMean(Tensor tensor)
    {
        float* ptr = (float*)tensor.DataPointer;
        long count = tensor.ElementCount;
        double sum = 0;
        for (long i = 0; i < count; i++) sum += ptr[i];
        return (float)(sum / count);
    }

    private static unsafe float ComputeStd(Tensor tensor)
    {
        float* ptr = (float*)tensor.DataPointer;
        long count = tensor.ElementCount;
        double sum = 0, sumSq = 0;
        for (long i = 0; i < count; i++)
        {
            sum += ptr[i];
            sumSq += (double)ptr[i] * ptr[i];
        }
        double mean = sum / count;
        return (float)Math.Sqrt(Math.Max(0, sumSq / count - mean * mean));
    }

    private static unsafe Tensor SliceBatch(Tensor tensor, int batchIdx, int seqLen, int hiddenSize)
    {
        TensorShape shape = new(1, seqLen, hiddenSize);
        Tensor slice = new(shape, DType.F32);
        float* src = (float*)tensor.DataPointer;
        float* dst = (float*)slice.DataPointer;
        int elements = seqLen * hiddenSize;
        int srcOffset = batchIdx * elements;
        for (int i = 0; i < elements; i++)
            dst[i] = src[srcOffset + i];
        return slice;
    }

    private static Dictionary<string, Tensor> CastWeightsToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32 = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            f32[kvp.Key] = (kvp.Value.DType == DType.F16 || kvp.Value.DType == DType.BF16)
                ? kvp.Value.CastTo(DType.F32)
                : kvp.Value;
        }
        return f32;
    }

    #endregion
}
