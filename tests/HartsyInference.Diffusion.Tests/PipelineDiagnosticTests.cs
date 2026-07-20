using System.Text.Json;
using Xunit;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>
/// Diagnostic test that dumps intermediate tensor statistics at key pipeline stages.
/// Compare output against Python reference (dump_reference_stats.py) to find divergences.
/// </summary>
public class PipelineDiagnosticTests
{
    private static string ModelDir => TestPaths.Sd15.DiffusersDir;

    private static readonly JsonSerializerOptions _jsonOpts = new() { WriteIndented = true };

    [Fact(Skip = "Manual diagnostic — run explicitly with: dotnet test --filter DumpEarlyStageStats")]
    public unsafe void DumpEarlyStageStats()
    {
        if (!Directory.Exists(ModelDir))
        {
            Console.WriteLine($"SKIPPED: Model directory not found: {ModelDir}");
            return;
        }

        using CpuBackend backend = new();

        // ── 1. Tokenize ──
        string tokenizerVocab = Path.Combine(ModelDir, "tokenizer", "vocab.json");
        string tokenizerMerges = Path.Combine(ModelDir, "tokenizer", "merges.txt");
        using ClipTokenizer tokenizer = new(tokenizerVocab, tokenizerMerges);

        string prompt = "a painting of a cat sitting on a windowsill";
        string negativePrompt = "blurry, bad quality";
        int[] promptTokens = tokenizer.Encode(prompt);
        int[] negativeTokens = tokenizer.Encode(negativePrompt);

        Console.WriteLine($"PROMPT_TOKENS: [{string.Join(", ", promptTokens)}]");
        Console.WriteLine($"NEGATIVE_TOKENS: [{string.Join(", ", negativeTokens)}]");
        Console.WriteLine($"Prompt token count: {promptTokens.Length}");
        Console.WriteLine($"Negative token count: {negativeTokens.Length}");

        // ── 2. Text Encoding ──
        Console.WriteLine("Loading text encoder...");
        ClipTextEncoderConfig clipConfig = ClipTextEncoderConfig.Sd15;
        ClipTextEncoder textEncoder = new(clipConfig);

        using SafeTensorsLoader textEncoderLoader = new();
        textEncoderLoader.Load(Path.Combine(ModelDir, "text_encoder", "model.fp16.safetensors"));
        Dictionary<string, Tensor> textEncoderWeights = CastWeightsToF32(textEncoderLoader.GetAllTensors());

        // Print a few weight keys and stats
        int printCount = 0;
        foreach (KeyValuePair<string, Tensor> kvp in textEncoderWeights)
        {
            if (printCount++ < 5)
                Console.WriteLine($"  Weight: {kvp.Key} shape=[{string.Join(",", Enumerable.Range(0, kvp.Value.Shape.Rank).Select(d => kvp.Value.Shape[d]))}] dtype={kvp.Value.DType}");
        }
        Console.WriteLine($"  Total weights: {textEncoderWeights.Count}");

        textEncoder.LoadWeights(textEncoderWeights, "text_model");

        Console.WriteLine("Encoding text...");
        int[][] batchTokenIds = [negativeTokens, promptTokens];
        Tensor textEmbeddings = textEncoder.Encode(backend, batchTokenIds);

        int seqLen = (int)textEmbeddings.Shape[1];
        int hiddenSize = (int)textEmbeddings.Shape[2];

        Tensor negEmb = SliceBatch(textEmbeddings, 0, seqLen, hiddenSize);
        Tensor posEmb = SliceBatch(textEmbeddings, 1, seqLen, hiddenSize);

        Console.WriteLine($"NEG_EMB: mean={ComputeMean(negEmb):F6}, std={ComputeStd(negEmb):F6}, first8=[{string.Join(", ", GetFirst8(negEmb).Select(v => v.ToString("F6")))}]");
        Console.WriteLine($"POS_EMB: mean={ComputeMean(posEmb):F6}, std={ComputeStd(posEmb):F6}, first8=[{string.Join(", ", GetFirst8(posEmb).Select(v => v.ToString("F6")))}]");
        Console.WriteLine($"CONCAT_EMB: mean={ComputeMean(textEmbeddings):F6}, std={ComputeStd(textEmbeddings):F6}");

        negEmb.Dispose();
        posEmb.Dispose();

        // ── 3. Noise ──
        int width = 256, height = 256, steps = 20, seed = 42;
        int latentH = height / 8, latentW = width / 8;
        TensorShape latentShape = new(1, 4, latentH, latentW);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);
        Console.WriteLine($"NOISE: mean={ComputeMean(latent):F6}, std={ComputeStd(latent):F6}, first8=[{string.Join(", ", GetFirst8(latent).Select(v => v.ToString("F6")))}]");

        // ── 4. Scheduler ──
        EulerDiscreteScheduler scheduler = new();
        scheduler.SetTimesteps(steps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        float[] tsArr = new float[steps];
        timesteps.CopyTo(tsArr);
        Console.WriteLine($"TIMESTEPS: [{string.Join(", ", tsArr.Select(t => t.ToString("F4")))}]");
        Console.WriteLine($"INIT_NOISE_SIGMA: {scheduler.InitialNoiseSigma:F6}");

        // Scale initial noise
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new(latentShape, DType.F32);
            backend.Scale(scaled, latent, initSigma);
            latent.Dispose();
            latent = scaled;
        }
        Console.WriteLine($"SCALED_NOISE: mean={ComputeMean(latent):F6}, std={ComputeStd(latent):F6}, first8=[{string.Join(", ", GetFirst8(latent).Select(v => v.ToString("F6")))}]");

        latent.Dispose();
        textEmbeddings.Dispose();
    }

    private static unsafe float[] GetFirst8(Tensor tensor)
    {
        float* ptr = (float*)tensor.DataPointer;
        int count = Math.Min(8, (int)tensor.ElementCount);
        float[] result = new float[count];
        for (int i = 0; i < count; i++) result[i] = ptr[i];
        return result;
    }

    [Fact(Skip = "Manual diagnostic — run explicitly with: dotnet test --filter DumpPipelineStats")]
    public unsafe void DumpPipelineStats()
    {
        string outputPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
            "hartsyinference_diagnostic_stats.json");

        if (!Directory.Exists(ModelDir))
        {
            Console.WriteLine($"SKIPPED: Model directory not found: {ModelDir}");
            return;
        }

        List<Dictionary<string, object>> stats = [];
        using CpuBackend backend = new();

        // ── 1. Tokenize ──
        string tokenizerVocab = Path.Combine(ModelDir, "tokenizer", "vocab.json");
        string tokenizerMerges = Path.Combine(ModelDir, "tokenizer", "merges.txt");
        using ClipTokenizer tokenizer = new(tokenizerVocab, tokenizerMerges);

        string prompt = "a painting of a cat sitting on a windowsill";
        string negativePrompt = "blurry, bad quality";
        int[] promptTokens = tokenizer.Encode(prompt);
        int[] negativeTokens = tokenizer.Encode(negativePrompt);

        stats.Add(new Dictionary<string, object>
        {
            ["name"] = "prompt_token_ids",
            ["values"] = promptTokens.Select(x => (object)x).ToArray()
        });
        stats.Add(new Dictionary<string, object>
        {
            ["name"] = "negative_token_ids",
            ["values"] = negativeTokens.Select(x => (object)x).ToArray()
        });

        // ── 2. Text Encoding ──
        Console.WriteLine("Loading text encoder...");
        ClipTextEncoderConfig clipConfig = ClipTextEncoderConfig.Sd15;
        ClipTextEncoder textEncoder = new(clipConfig);

        using SafeTensorsLoader textEncoderLoader = new();
        textEncoderLoader.Load(Path.Combine(ModelDir, "text_encoder", "model.fp16.safetensors"));
        Dictionary<string, Tensor> textEncoderWeights = CastWeightsToF32(textEncoderLoader.GetAllTensors());
        textEncoder.LoadWeights(textEncoderWeights, "text_model");

        Console.WriteLine("Encoding text...");
        int[][] batchTokenIds = [negativeTokens, promptTokens];
        Tensor textEmbeddings = textEncoder.Encode(backend, batchTokenIds);

        // Split to get individual embeddings for comparison
        int seqLen = (int)textEmbeddings.Shape[1];
        int hiddenSize = (int)textEmbeddings.Shape[2];

        Tensor negEmb = SliceBatch(textEmbeddings, 0, seqLen, hiddenSize);
        Tensor posEmb = SliceBatch(textEmbeddings, 1, seqLen, hiddenSize);

        stats.Add(TensorStats("negative_embeddings", negEmb));
        stats.Add(TensorStats("prompt_embeddings", posEmb));
        stats.Add(TensorStats("text_embeddings_concat", textEmbeddings));
        negEmb.Dispose();
        posEmb.Dispose();

        // ── 3. Noise ──
        Console.WriteLine("Generating noise...");
        int width = 256, height = 256, steps = 20, seed = 42;
        float cfgScale = 7.5f;
        int latentH = height / 8, latentW = width / 8;
        TensorShape latentShape = new(1, 4, latentH, latentW);

        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);
        stats.Add(TensorStats("initial_noise", latent));

        // ── 4. Scheduler ──
        EulerDiscreteScheduler scheduler = new();
        scheduler.SetTimesteps(steps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        float[] timestepValues = new float[steps];
        timesteps.CopyTo(timestepValues);
        stats.Add(new Dictionary<string, object>
        {
            ["name"] = "timesteps",
            ["values"] = timestepValues.Select(x => (object)x).ToArray()
        });
        stats.Add(new Dictionary<string, object>
        {
            ["name"] = "initial_noise_sigma",
            ["value"] = scheduler.InitialNoiseSigma
        });

        // Scale initial noise
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new(latentShape, DType.F32);
            backend.Scale(scaled, latent, initSigma);
            latent.Dispose();
            latent = scaled;
        }
        stats.Add(TensorStats("scaled_noise", latent));

        // ── 5. Load UNet ──
        Console.WriteLine("Loading UNet...");
        UNetConfig unetConfig = UNetConfig.Sd15;
        UNet unet = new(unetConfig);

        using SafeTensorsLoader unetLoader = new();
        unetLoader.Load(Path.Combine(ModelDir, "unet", "diffusion_pytorch_model.fp16.safetensors"));
        Dictionary<string, Tensor> unetWeights = CastWeightsToF32(unetLoader.GetAllTensors());
        unet.LoadWeights(unetWeights);

        // ── 6. Denoising (first 3 steps with diagnostics) ──
        Console.WriteLine("Running denoising steps...");
        int diagSteps = Math.Min(3, steps);

        for (int i = 0; i < diagSteps; i++)
        {
            float t = timesteps[i];

            // Scale model input
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
            stats.Add(TensorStats($"step{i}_scaled_input", scaledLatent));

            // Run UNet for unconditional and conditional
            Tensor uncondEmb = SliceBatch(textEmbeddings, 0, seqLen, hiddenSize);
            Tensor condEmb = SliceBatch(textEmbeddings, 1, seqLen, hiddenSize);

            Console.WriteLine($"  Step {i}: t={t:F1}, running UNet (uncond)...");
            Tensor uncondNoise = unet.Forward(backend, scaledLatent, t, uncondEmb);
            stats.Add(TensorStats($"step{i}_noise_pred_uncond", uncondNoise));

            Console.WriteLine($"  Step {i}: t={t:F1}, running UNet (cond)...");
            Tensor condNoise = unet.Forward(backend, scaledLatent, t, condEmb);
            stats.Add(TensorStats($"step{i}_noise_pred_cond", condNoise));

            uncondEmb.Dispose();
            condEmb.Dispose();

            // CFG
            Tensor noisePredCfg = new(latentShape, DType.F32);
            float* uncPtr = (float*)uncondNoise.DataPointer;
            float* conPtr = (float*)condNoise.DataPointer;
            float* cfgPtr = (float*)noisePredCfg.DataPointer;
            int count = (int)latentShape.ElementCount;
            for (int j = 0; j < count; j++)
            {
                cfgPtr[j] = uncPtr[j] + cfgScale * (conPtr[j] - uncPtr[j]);
            }
            uncondNoise.Dispose();
            condNoise.Dispose();

            stats.Add(TensorStats($"step{i}_noise_pred_cfg", noisePredCfg));

            if (scaledLatent != latent) scaledLatent.Dispose();

            // Scheduler step
            Tensor newLatent = new(latentShape, DType.F32);
            scheduler.Step(newLatent, noisePredCfg, latent, i);
            noisePredCfg.Dispose();
            latent.Dispose();
            latent = newLatent;

            stats.Add(TensorStats($"step{i}_latents_after", latent));
            Console.WriteLine($"  Step {i}: latent mean={ComputeMean(latent):F6}, std={ComputeStd(latent):F6}");
        }

        // Run remaining steps without diagnostics
        for (int i = diagSteps; i < steps; i++)
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

            Tensor noisePredCfg = new(latentShape, DType.F32);
            float* up = (float*)uncondNoise.DataPointer;
            float* cp = (float*)condNoise.DataPointer;
            float* fp = (float*)noisePredCfg.DataPointer;
            int cnt = (int)latentShape.ElementCount;
            for (int j = 0; j < cnt; j++)
            {
                fp[j] = up[j] + cfgScale * (cp[j] - up[j]);
            }
            uncondNoise.Dispose();
            condNoise.Dispose();

            if (scaledLatent != latent) scaledLatent.Dispose();

            Tensor newLatent = new(latentShape, DType.F32);
            scheduler.Step(newLatent, noisePredCfg, latent, i);
            noisePredCfg.Dispose();
            latent.Dispose();
            latent = newLatent;

            if (i % 5 == 0)
                Console.WriteLine($"  Step {i}: t={t:F1}, latent mean={ComputeMean(latent):F6}");
        }

        stats.Add(TensorStats("final_latents", latent));
        textEmbeddings.Dispose();

        // ── 7. VAE decode ──
        Console.WriteLine("Loading VAE...");
        VaeConfig vaeConfig = VaeConfig.Sd15;
        VaeDecoder vaeDecoder = new(vaeConfig);

        using SafeTensorsLoader vaeLoader = new();
        vaeLoader.Load(Path.Combine(ModelDir, "vae", "diffusion_pytorch_model.fp16.safetensors"));
        Dictionary<string, Tensor> vaeWeights = CastWeightsToF32(vaeLoader.GetAllTensors());
        vaeDecoder.LoadWeights(vaeWeights);

        Console.WriteLine("Decoding with VAE...");
        Tensor image = vaeDecoder.Decode(backend, latent);
        latent.Dispose();

        stats.Add(TensorStats("vae_output", image));
        image.Dispose();

        // Save
        string json = JsonSerializer.Serialize(stats, _jsonOpts);
        File.WriteAllText(outputPath, json);
        Console.WriteLine($"\nStats saved to {outputPath}");
        Console.WriteLine($"Total stats entries: {stats.Count}");
    }

    private static unsafe Dictionary<string, object> TensorStats(string name, Tensor tensor)
    {
        float* ptr = (float*)tensor.DataPointer;
        long count = tensor.ElementCount;

        double sum = 0, sumSq = 0;
        float min = float.MaxValue, max = float.MinValue;
        float[] first8 = new float[Math.Min(8, (int)count)];

        for (long i = 0; i < count; i++)
        {
            float v = ptr[i];
            sum += v;
            sumSq += (double)v * v;
            if (v < min) min = v;
            if (v > max) max = v;
            if (i < 8) first8[i] = v;
        }

        double mean = sum / count;
        double variance = sumSq / count - mean * mean;
        double std = Math.Sqrt(Math.Max(0, variance));

        long[] shape = new long[tensor.Shape.Rank];
        for (int d = 0; d < tensor.Shape.Rank; d++)
            shape[d] = tensor.Shape[d];

        return new Dictionary<string, object>
        {
            ["name"] = name,
            ["shape"] = shape,
            ["mean"] = mean,
            ["std"] = std,
            ["min"] = (double)min,
            ["max"] = (double)max,
            ["abs_mean"] = Math.Abs(sum) / count, // rough
            ["first_8"] = first8.Select(x => (object)(double)x).ToArray()
        };
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

    /// <summary>Runs full pipeline and saves latent preview images at multiple steps plus final VAE output.
    /// This tells us whether the problem is in denoising or VAE decode.</summary>
    [Fact]
    public unsafe void LatentPreviewDiagnostic()
    {
        if (!Directory.Exists(ModelDir))
        {
            Console.WriteLine($"SKIPPED: Model directory not found: {ModelDir}");
            return;
        }

        string outputDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "latent_debug");
        Directory.CreateDirectory(outputDir);

        using CpuBackend backend = new();

        // ── Load all models ──
        string tokenizerVocab = Path.Combine(ModelDir, "tokenizer", "vocab.json");
        string tokenizerMerges = Path.Combine(ModelDir, "tokenizer", "merges.txt");
        using ClipTokenizer tokenizer = new(tokenizerVocab, tokenizerMerges);

        string prompt = "a painting of a cat sitting on a windowsill";
        string negativePrompt = "blurry, bad quality";
        int[] promptTokens = tokenizer.Encode(prompt);
        int[] negativeTokens = tokenizer.Encode(negativePrompt);

        ClipTextEncoderConfig clipConfig = ClipTextEncoderConfig.Sd15;
        ClipTextEncoder textEncoder = new(clipConfig);
        using SafeTensorsLoader teLoader = new();
        teLoader.Load(Path.Combine(ModelDir, "text_encoder", "model.fp16.safetensors"));
        Dictionary<string, Tensor> teWeights = CastWeightsToF32(teLoader.GetAllTensors());
        textEncoder.LoadWeights(teWeights, "text_model");

        UNet unet = new(UNetConfig.Sd15);
        using SafeTensorsLoader unetLoader = new();
        unetLoader.Load(Path.Combine(ModelDir, "unet", "diffusion_pytorch_model.fp16.safetensors"));
        Dictionary<string, Tensor> unetWeights = CastWeightsToF32(unetLoader.GetAllTensors());
        unet.LoadWeights(unetWeights);

        VaeDecoder vaeDecoder = new(VaeConfig.Sd15);
        using SafeTensorsLoader vaeLoader = new();
        vaeLoader.Load(Path.Combine(ModelDir, "vae", "diffusion_pytorch_model.fp16.safetensors"));
        Dictionary<string, Tensor> vaeWeights = CastWeightsToF32(vaeLoader.GetAllTensors());
        vaeDecoder.LoadWeights(vaeWeights);

        Console.WriteLine("All models loaded.");

        // ── Settings ──
        int width = 256, height = 256, steps = 20, seed = 42;
        float cfgScale = 7.5f;
        int latentH = height / 8, latentW = width / 8;
        TensorShape latentShape = new(1, 4, latentH, latentW);

        // ── Text encode ──
        int[][] batchTokenIds = [negativeTokens, promptTokens];
        Tensor textEmbeddings = textEncoder.Encode(backend, batchTokenIds);
        int seqLen = (int)textEmbeddings.Shape[1];
        int hiddenSize = (int)textEmbeddings.Shape[2];

        Console.WriteLine($"Text embeddings: mean={ComputeMean(textEmbeddings):F6}, std={ComputeStd(textEmbeddings):F6}");

        // ── Noise + scheduler ──
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);
        EulerDiscreteScheduler scheduler = new();
        scheduler.SetTimesteps(steps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new(latentShape, DType.F32);
            backend.Scale(scaled, latent, initSigma);
            latent.Dispose();
            latent = scaled;
        }

        // Save initial noise as latent preview
        SaveLatentPreview(latent, latentH, latentW, Path.Combine(outputDir, "00_initial_noise.bmp"));

        // ── Denoising loop ──
        int[] previewSteps = [0, 1, 2, 4, 9, 14, 19]; // steps to save preview at

        for (int i = 0; i < steps; i++)
        {
            System.Diagnostics.Stopwatch stepSw = System.Diagnostics.Stopwatch.StartNew();
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

            // CFG: two UNet passes
            Tensor uncondEmb = SliceBatch(textEmbeddings, 0, seqLen, hiddenSize);
            Tensor condEmb = SliceBatch(textEmbeddings, 1, seqLen, hiddenSize);

            Tensor uncondNoise = unet.Forward(backend, scaledLatent, t, uncondEmb);
            Tensor condNoise = unet.Forward(backend, scaledLatent, t, condEmb);
            uncondEmb.Dispose();
            condEmb.Dispose();

            // CFG combine
            Tensor noisePred = new(latentShape, DType.F32);
            float* up = (float*)uncondNoise.DataPointer;
            float* cp = (float*)condNoise.DataPointer;
            float* fp = (float*)noisePred.DataPointer;
            int count = (int)latentShape.ElementCount;
            for (int j = 0; j < count; j++)
            {
                fp[j] = up[j] + cfgScale * (cp[j] - up[j]);
            }
            uncondNoise.Dispose();
            condNoise.Dispose();

            if (scaledLatent != latent) scaledLatent.Dispose();

            Tensor newLatent = new(latentShape, DType.F32);
            scheduler.Step(newLatent, noisePred, latent, i);
            noisePred.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            Console.WriteLine($"Step {i + 1}/{steps} (t={t:F1}): mean={ComputeMean(latent):F6}, std={ComputeStd(latent):F6}, {stepSw.ElapsedMilliseconds}ms");

            // Save latent preview at selected steps
            if (Array.IndexOf(previewSteps, i) >= 0)
            {
                SaveLatentPreview(latent, latentH, latentW,
                    Path.Combine(outputDir, $"{i + 1:D2}_step{i + 1}_latent.bmp"));
            }
        }

        textEmbeddings.Dispose();

        // ── Save final latent stats ──
        Console.WriteLine($"\nFinal latents: mean={ComputeMean(latent):F6}, std={ComputeStd(latent):F6}");
        Console.WriteLine($"Final latents min/max: {ComputeMin(latent):F6} / {ComputeMax(latent):F6}");

        // Save per-channel stats for the 4 latent channels
        for (int ch = 0; ch < 4; ch++)
        {
            float* ptr = (float*)latent.DataPointer;
            int spatial = latentH * latentW;
            double sum = 0;
            float cMin = float.MaxValue, cMax = float.MinValue;
            for (int j = 0; j < spatial; j++)
            {
                float v = ptr[ch * spatial + j];
                sum += v;
                if (v < cMin) cMin = v;
                if (v > cMax) cMax = v;
            }
            Console.WriteLine($"  Channel {ch}: mean={sum / spatial:F6}, min={cMin:F6}, max={cMax:F6}");
        }

        // ── VAE decode ──
        Console.WriteLine("\nDecoding with VAE...");
        System.Diagnostics.Stopwatch vaeSw = System.Diagnostics.Stopwatch.StartNew();
        Tensor image = vaeDecoder.Decode(backend, latent);
        vaeSw.Stop();
        Console.WriteLine($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");
        Console.WriteLine($"VAE output: mean={ComputeMean(image):F6}, std={ComputeStd(image):F6}");
        Console.WriteLine($"VAE output min/max: {ComputeMin(image):F6} / {ComputeMax(image):F6}");

        // Save per-channel stats for RGB
        for (int ch = 0; ch < 3; ch++)
        {
            float* ptr = (float*)image.DataPointer;
            int spatial = height * width;
            double sum = 0;
            float cMin = float.MaxValue, cMax = float.MinValue;
            for (int j = 0; j < spatial; j++)
            {
                float v = ptr[ch * spatial + j];
                sum += v;
                if (v < cMin) cMin = v;
                if (v > cMax) cMax = v;
            }
            string chName = ch == 0 ? "R" : ch == 1 ? "G" : "B";
            Console.WriteLine($"  {chName}: mean={sum / spatial:F6}, min={cMin:F6}, max={cMax:F6}");
        }

        // Save final image
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        string finalPath = Path.Combine(outputDir, "final_output.bmp");
        ImagePostProcessor.SaveBmp(finalPath, rgbData, width, height);
        image.Dispose();
        latent.Dispose();

        Console.WriteLine($"\nAll outputs saved to: {outputDir}");
        Console.WriteLine("Files:");
        foreach (string file in Directory.GetFiles(outputDir, "*.bmp"))
        {
            Console.WriteLine($"  {Path.GetFileName(file)}");
        }
    }

    /// <summary>Saves a latent tensor [1, 4, H, W] as a grayscale BMP preview.
    /// Averages the 4 channels, then normalizes to [0, 255]. This gives a rough preview of what the denoiser is producing.</summary>
    private static unsafe void SaveLatentPreview(Tensor latent, int h, int w, string path)
    {
        float* ptr = (float*)latent.DataPointer;
        int spatial = h * w;

        // Average 4 channels into one grayscale image
        float[] gray = new float[spatial];
        for (int ch = 0; ch < 4; ch++)
        {
            for (int i = 0; i < spatial; i++)
            {
                gray[i] += ptr[ch * spatial + i];
            }
        }
        for (int i = 0; i < spatial; i++)
        {
            gray[i] /= 4.0f;
        }

        // Find min/max for normalization
        float min = float.MaxValue, max = float.MinValue;
        for (int i = 0; i < spatial; i++)
        {
            if (gray[i] < min) min = gray[i];
            if (gray[i] > max) max = gray[i];
        }

        float range = max - min;
        if (range < 1e-8f) range = 1.0f;

        // Convert to RGB bytes (grayscale = R=G=B)
        byte[] rgb = new byte[spatial * 3];
        for (int i = 0; i < spatial; i++)
        {
            byte val = (byte)Math.Clamp((gray[i] - min) / range * 255.0f + 0.5f, 0, 255);
            rgb[i * 3 + 0] = val;
            rgb[i * 3 + 1] = val;
            rgb[i * 3 + 2] = val;
        }

        // Also save a color version using first 3 channels as RGB
        // This is more informative — each latent channel maps to a color
        byte[] colorRgb = new byte[spatial * 3];
        float[] chMin = new float[3];
        float[] chMax = new float[3];
        for (int ch = 0; ch < 3; ch++)
        {
            chMin[ch] = float.MaxValue;
            chMax[ch] = float.MinValue;
            for (int i = 0; i < spatial; i++)
            {
                float v = ptr[ch * spatial + i];
                if (v < chMin[ch]) chMin[ch] = v;
                if (v > chMax[ch]) chMax[ch] = v;
            }
        }
        for (int i = 0; i < spatial; i++)
        {
            for (int ch = 0; ch < 3; ch++)
            {
                float v = ptr[ch * spatial + i];
                float r = chMax[ch] - chMin[ch];
                if (r < 1e-8f) r = 1.0f;
                byte val = (byte)Math.Clamp((v - chMin[ch]) / r * 255.0f + 0.5f, 0, 255);
                colorRgb[i * 3 + ch] = val;
            }
        }

        // Save both versions
        ImagePostProcessor.SaveBmp(path, rgb, w, h);

        string colorPath = Path.ChangeExtension(path, null) + "_color.bmp";
        ImagePostProcessor.SaveBmp(colorPath, colorRgb, w, h);
    }

    private static unsafe float ComputeMin(Tensor tensor)
    {
        float* ptr = (float*)tensor.DataPointer;
        long count = tensor.ElementCount;
        float min = float.MaxValue;
        for (long i = 0; i < count; i++)
        {
            if (ptr[i] < min) min = ptr[i];
        }
        return min;
    }

    private static unsafe float ComputeMax(Tensor tensor)
    {
        float* ptr = (float*)tensor.DataPointer;
        long count = tensor.ElementCount;
        float max = float.MinValue;
        for (long i = 0; i < count; i++)
        {
            if (ptr[i] > max) max = ptr[i];
        }
        return max;
    }

    private static Dictionary<string, Tensor> CastWeightsToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32 = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            if (kvp.Value.DType == DType.F16 || kvp.Value.DType == DType.BF16)
                f32[kvp.Key] = kvp.Value.CastTo(DType.F32);
            else
                f32[kvp.Key] = kvp.Value;
        }
        return f32;
    }
}
