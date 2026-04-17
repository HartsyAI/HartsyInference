using System.Text.Json;
using Xunit;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tokenizers;

namespace SharpInference.Diffusion.Tests;

/// <summary>
/// Diagnostic test that dumps intermediate tensor statistics at key pipeline stages.
/// Compare output against Python reference (dump_reference_stats.py) to find divergences.
/// </summary>
public class PipelineDiagnosticTests
{
    private const string ModelDir = @"C:\Users\AI Overlord\Desktop\Projects\SharpInference\tests\test-models\sd15";

    private static readonly JsonSerializerOptions JsonOpts = new() { WriteIndented = true };

    [Fact]
    public unsafe void DumpPipelineStats()
    {
        string outputPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
            "sharpinference_diagnostic_stats.json");

        if (!Directory.Exists(ModelDir))
        {
            Assert.Fail($"Model directory not found: {ModelDir}");
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
        string json = JsonSerializer.Serialize(stats, JsonOpts);
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
