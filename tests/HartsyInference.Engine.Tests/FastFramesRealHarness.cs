using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Video.Pipelines;
using HartsyInference.ModelAssets.Metadata;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>Local-only harness for the FastFrames S reference comparison. Each test does nothing unless its environment
/// variable is set, so it never runs in CI. Not committed.</summary>
public sealed class FastFramesRealHarness
{
    private const string Prompt = "a cat walking through a sunlit garden";

    /// <summary>Writes the generator_ema dict of the released pickle as F32 safetensors, keys unchanged (FSDP prefix kept),
    /// with no pickle execution: the envelope reader only parses the pickle's data.</summary>
    [Fact]
    public void ConvertGeneratorEma()
    {
        string? src = Environment.GetEnvironmentVariable("FF_GEN_PT");
        string? dst = Environment.GetEnvironmentVariable("FF_GEN_OUT");
        if (src is null || dst is null) return;
        using PytorchPickleLoader pickle = new PytorchPickleLoader();
        pickle.OpenLazy(src, recursiveFlatten: true);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        const string head = "generator_ema.";
        foreach (PytorchTensorDescriptor d in pickle.Descriptors.Values
                     .Where(d => d.Name.StartsWith(head, StringComparison.Ordinal))
                     .OrderBy(d => d.StorageKey, StringComparer.Ordinal).ThenBy(d => d.StorageByteOffset))
        {
            weights[d.Name[head.Length..]] = pickle.ReadTensor(d.Name, DType.F32);
        }
        SafeTensorsWriter.Save(dst, weights);
        Assert.NotEmpty(weights);
        foreach (Tensor t in weights.Values) t.Dispose();
    }

    /// <summary>Runs the engine's S rollout on CUDA with the upstream dump's noise and prompt embedding, and writes the per-frame
    /// relative RMS of latents and pixels against the upstream FP32 CPU run.</summary>
    [Fact]
    public void RolloutMatchesUpstreamDump()
    {
        string? gen = Environment.GetEnvironmentVariable("FF_GEN_SAFE");
        string? io = Environment.GetEnvironmentVariable("FF_IO");
        string? embeds = Environment.GetEnvironmentVariable("FF_EMBEDS");
        string? report = Environment.GetEnvironmentVariable("FF_REPORT");
        string? kv = Environment.GetEnvironmentVariable("FF_KV");
        string? dec = Environment.GetEnvironmentVariable("FF_DEC");
        if (gen is null || io is null || embeds is null || report is null || kv is null || dec is null) return;

        bool onCpu = Environment.GetEnvironmentVariable("FF_CPU") == "1";
        using IBackend backend = onCpu ? new CpuBackend() : BackendFactory.CreateCuda(0);
        List<IDisposable> loaders = new List<IDisposable>();
        try
        {
            // The same reads the recipe does, on this backend: F16 generator from the released pickle.
            Dictionary<string, Tensor> generator = FastFramesRecipe.ReadGenerator(gen, onCpu ? DType.F32 : DType.F16, castForCpu: false, loaders);
            WanVideoCheckpointConverter.ConvertedWeights converted = WanVideoCheckpointConverter.Convert(generator);
            HartsyInference.Diffusion.Models.Denoisers.WanVideoConfig wan = HartsyInference.Diffusion.Models.Denoisers.WanConfigDetector.Detect(converted.Transformer);
            HartsyInference.Diffusion.Models.Denoisers.WanVideoTransformer transformer = new(wan);
            loaders.Add(transformer);
            transformer.LoadWeights(converted.Transformer);

            CheckpointSource kvSource = CheckpointSource.Open(kv);
            loaders.Add(kvSource);
            HartsyInference.Diffusion.Models.FastFrames.FastFramesKvRefiner refiner = new(
                HartsyInference.Diffusion.Models.FastFrames.FastFramesKvRefiner.DetectConfig(kvSource.Weights, wan.NumHeads));
            loaders.Add(refiner);
            refiner.LoadWeights(kvSource.Weights);

            CheckpointSource decSource = CheckpointSource.Open(dec);
            loaders.Add(decSource);
            HartsyInference.Diffusion.Models.FastFrames.FastFramesPixelDecoder decoder = new(
                HartsyInference.Diffusion.Models.FastFrames.FastFramesPixelDecoder.DetectConfig(decSource.Weights));
            loaders.Add(decoder);
            decoder.LoadWeights(decSource.Weights);

            HartsyInference.Video.Pipelines.FastFramesPipeline pipeline = new(backend, transformer, refiner, decoder,
                HartsyInference.Video.Pipelines.FastFramesConfig.S);

            CheckpointSource ioSource = CheckpointSource.Open(io);
            loaders.Add(ioSource);
            Tensor noise = ioSource.Weights["noise"].CastTo(DType.F32);
            Tensor transition = ioSource.Weights["transition"].CastTo(DType.F32);
            Tensor upstreamLatents = ioSource.Weights["latents"].CastTo(DType.F32);
            Tensor upstreamPixels = ioSource.Weights["pixels"].CastTo(DType.F32);
            CheckpointSource embedSource = CheckpointSource.Open(embeds);
            loaders.Add(embedSource);
            Tensor promptEmbeds = embedSource.Weights["prompt_embeds"].CastTo(DType.F32).Reshape(new TensorShape(512, 4096));
            int frames = (int)noise.Shape[0];
            int latentValues = 16 * 60 * 104;
            SavedNoise source = new(noise, transition, frames);

            List<string> lines = [];
            int f = 0;
            float[] refLat = Floats(upstreamLatents);
            float[] refPix = Floats(upstreamPixels);
            int pixelValues = 3 * 480 * 832;
            System.Diagnostics.Stopwatch clock = System.Diagnostics.Stopwatch.StartNew();
            foreach (HartsyInference.Video.Pipelines.FastFramesFrame frame in pipeline.Rollout(promptEmbeds, source, frames))
            {
                double latRel = Rel(refLat.AsSpan(frame.Index * latentValues, latentValues), frame.Latent);
                // Our latent frame f decodes pixel frames 4f..4f+3; upstream keeps decoded frames 3.. (its align drops the
                // first three), so decoded d is upstream frame d-3. Compare every pixel frame that exists upstream.
                double pixNum = 0, pixDen = 0;
                for (int k = 0; k < 4; k++)
                {
                    int u = frame.Index * 4 + k - 3;
                    if (u < 0 || u >= upstreamPixels.Shape[0]) continue;
                    ReadOnlySpan<float> ours = frame.Pixels.AsSpan(k * pixelValues, pixelValues);
                    ReadOnlySpan<float> theirs = refPix.AsSpan(u * pixelValues, pixelValues);
                    for (int i = 0; i < pixelValues; i++)
                    {
                        double d = ours[i] - theirs[i];
                        pixNum += d * d;
                        pixDen += (double)theirs[i] * theirs[i];
                    }
                }
                double pixRel = Math.Sqrt(pixNum / Math.Max(pixDen, 1e-30));
                lines.Add($"latent frame {frame.Index}: rel RMS {latRel:G6}; pixels rel RMS {pixRel:G6} (pixel frames {frame.PixelFrames}, {clock.Elapsed.TotalSeconds:F1} s)");
                lines.Add($"  ours mean/std {MeanStd(frame.Pixels)}; upstream mean/std {MeanStd(refPix.AsSpan(Math.Max(0, frame.Index * 4 - 3) * pixelValues, 4 * pixelValues))}");
                f++;
            }
            File.WriteAllLines(report, lines);
            Assert.Equal(frames, f);
        }
        finally
        {
            foreach (IDisposable l in loaders) l.Dispose();
        }
    }

    private sealed class SavedNoise(Tensor initial, Tensor transition, int frames) : IFastFramesNoise
    {
        private const int Values = 16 * 60 * 104;
        public float[] Initial(int frame) => Copy(initial, frame * Values);
        public float[] Transition(int step, int frame) =>
            Copy(transition, (step * frames + frame) * Values);
    }

    internal static string MeanStd(ReadOnlySpan<float> v)
    {
        double m = 0; foreach (float x in v) m += x; m /= v.Length;
        double s = 0; foreach (float x in v) s += (x - m) * (x - m);
        return $"{m:F4}/{Math.Sqrt(s / v.Length):F4}";
    }

    internal static string Dims(Tensor t) => string.Join("x", Enumerable.Range(0, t.Shape.Rank).Select(i => t.Shape[i]));

    internal static float[] Floats(Tensor t)
    {
        float[] values = new float[t.Shape.ElementCount];
        t.AsSpan<float>().CopyTo(values);
        return values;
    }

    private static float[] Copy(Tensor t, int offset)
    {
        float[] values = new float[16 * 60 * 104];
        t.AsSpan<float>().Slice(offset, values.Length).CopyTo(values);
        return values;
    }

    internal static double Rel(ReadOnlySpan<float> reference, ReadOnlySpan<float> actual)
    {
        double num = 0, den = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double d = actual[i] - reference[i];
            num += d * d;
            den += (double)reference[i] * reference[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    /// <summary>Encodes the prompt with the engine's umT5 (the fp8 file the recipe uses) and writes the 512-row embedding.</summary>
    [Fact]
    public void EncodePrompt()
    {
        string? umt5 = Environment.GetEnvironmentVariable("FF_UMT5");
        string? dst = Environment.GetEnvironmentVariable("FF_EMBED_OUT");
        if (umt5 is null || dst is null) return;
        using CpuBackend backend = new CpuBackend();
        List<IDisposable> loaders = new List<IDisposable>();
        (T5TextEncoder encoder, T5Tokenizer tokenizer) = VideoRecipeUtils.LoadUmt5(umt5, loaders);
        int[] tokens = tokenizer.Encode(PromptTagFlattening.Flatten(Prompt));
        Tensor[] embeds = VideoRecipeUtils.EncodeWanPromptBatch(backend, encoder, 4096, tokens);
        SafeTensorsWriter.Save(dst, new Dictionary<string, Tensor> { ["prompt_embeds"] = embeds[0] });
        foreach (Tensor t in embeds) t.Dispose();
        foreach (IDisposable l in loaders) l.Dispose();
    }
}

public sealed class FastFramesDecoderBackendHarness
{
    /// <summary>The real S pixel decoder on CUDA and on CPU, with one shared random hidden state and time embedding: writes the
    /// relative difference and each output's mean and standard deviation.</summary>
    [Fact]
    public void DecoderCudaMatchesCpu()
    {
        string? dec = Environment.GetEnvironmentVariable("FF_DEC");
        string? report = Environment.GetEnvironmentVariable("FF_DEC_REPORT");
        if (dec is null || report is null) return;
        CheckpointSource source = CheckpointSource.Open(dec);
        try
        {
            HartsyInference.Diffusion.Models.FastFrames.FastFramesPixelDecoderConfig config = HartsyInference.Diffusion.Models.FastFrames.FastFramesPixelDecoder.DetectConfig(source.Weights);
            Random rng = new(7);
            float[] hiddenValues = new float[1560 * config.InputDim];
            for (int i = 0; i < hiddenValues.Length; i++) hiddenValues[i] = (float)(rng.NextDouble() * 2 - 1);
            float[] tembValues = new float[config.InputDim];
            for (int i = 0; i < tembValues.Length; i++) tembValues[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
            List<string> lines = [];
            float[]? cpuOut = null;
            foreach (bool onCpu in new[] { true, false })
            {
                using IBackend backend = onCpu ? new CpuBackend() : BackendFactory.CreateCuda(0);
                HartsyInference.Diffusion.Models.FastFrames.FastFramesPixelDecoder decoder = new(config);
                try
                {
                    decoder.LoadWeights(source.Weights);
                    using Tensor hidden = new Tensor(new TensorShape(1560, config.InputDim), DType.F32);
                    hiddenValues.AsSpan().CopyTo(hidden.AsSpan<float>());
                    using Tensor temb = new Tensor(new TensorShape(1, config.InputDim), DType.F32);
                    tembValues.AsSpan().CopyTo(temb.AsSpan<float>());
                    using HartsyInference.Diffusion.Models.FastFrames.FastFramesPixelDecoder.State state = decoder.CreateState();
                    using Tensor pixels = decoder.DecodeFrame(backend, hidden, temb, 30, 52, state);
                    float[] values = FastFramesRealHarness.Floats(pixels);
                    lines.Add($"{(onCpu ? "cpu" : "cuda")}: shape {FastFramesRealHarness.Dims(pixels)} mean/std {FastFramesRealHarness.MeanStd(values)}");
                    if (onCpu) cpuOut = values;
                    else lines.Add($"relative difference cuda vs cpu: {FastFramesRealHarness.Rel(cpuOut!, values):G6}");
                }
                finally
                {
                    decoder.Dispose();
                }
            }
            File.WriteAllLines(report, lines);
        }
        finally
        {
            source.Dispose();
        }
    }
}

public sealed class FastFramesOpsBackendHarness
{
    /// <summary>The backend ops the pixel decoder uses, on CUDA and on CPU with one shared random input: relative difference.</summary>
    [Fact]
    public void OpsCudaMatchCpu()
    {
        string? report = Environment.GetEnvironmentVariable("FF_OPS_REPORT");
        if (report is null) return;
        Random rng = new(11);
        float[] tokenMajor = RandomValues(rng, 1560 * 1536);
        float[] weights = RandomValues(rng, 1536 * 9);
        List<string> lines = [];
        foreach (string op in new[] { "transpose", "depthwise_conv" })
        {
            float[]? cpu = null;
            foreach (bool onCpu in new[] { true, false })
            {
                using IBackend backend = onCpu ? new CpuBackend() : BackendFactory.CreateCuda(0);
                float[] result;
                if (op == "transpose")
                {
                    using Tensor input = new Tensor(new TensorShape(1, 1560, 1536), DType.F32);
                    tokenMajor.AsSpan().CopyTo(input.AsSpan<float>());
                    using Tensor output = new Tensor(new TensorShape(1, 1536, 1560), DType.F32);
                    backend.Transpose2D(output, input, 1560, 1536);
                    result = FastFramesRealHarness.Floats(output);
                }
                else
                {
                    using Tensor input = new Tensor(new TensorShape(1, 1536, 30, 52), DType.F32);
                    tokenMajor.AsSpan().CopyTo(input.AsSpan<float>());
                    using Tensor weight = new Tensor(new TensorShape(1536, 1, 3, 3), DType.F32);
                    weights.AsSpan().CopyTo(weight.AsSpan<float>());
                    using Tensor output = new Tensor(new TensorShape(1, 1536, 30, 52), DType.F32);
                    backend.Conv2dDepthwise(output, input, weight, null, 1, 1, 1, 1);
                    result = FastFramesRealHarness.Floats(output);
                }
                if (onCpu) cpu = result;
                else lines.Add($"{op}: relative difference cuda vs cpu {FastFramesRealHarness.Rel(cpu!, result):G6}");
            }
        }
        File.WriteAllLines(report, lines);
    }

    private static float[] RandomValues(Random rng, int count)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2 - 1);
        return values;
    }
}
