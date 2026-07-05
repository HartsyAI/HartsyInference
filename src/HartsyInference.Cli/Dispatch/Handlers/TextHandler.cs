using System.Diagnostics;
using System.Globalization;
using HartsyInference.Cli.Infra;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace HartsyInference.Cli.Dispatch.Handlers;

/// <summary>LLM text generation. Loads a GGUF checkpoint (any arch, config from metadata) or a safetensors directory
/// (Qwen2/Qwen3 presets) and streams tokens through the shared <see cref="TextGenerationPipeline"/>.</summary>
public sealed class TextHandler : IModalityHandler
{
    /// <inheritdoc/>
    public Modality Modality => Modality.Text;

    /// <inheritdoc/>
    public IModalityRunner Load(ModelSpec spec, IBackend backend, IProgressSink progress)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException(
                "No local weights found. Pass a .gguf file or safetensors directory via --model-path, " +
                "or place it under the models root.");
        }

        bool cuda = backend is CudaBackend;
        string path = spec.LocalPath;

        if (File.Exists(path) && path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            return LoadGguf(spec, path, backend, cuda, progress);

        if (Directory.Exists(path))
            return LoadSafetensors(spec, path, backend, cuda, progress);

        throw new FileNotFoundException($"Text model path is neither a .gguf file nor a directory: {path}");
    }

    /// <inheritdoc/>
    public GeneratedArtifact Run(IModalityRunner runner, string prompt, ParamState parameters, IProgressSink progress, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        TextRunner text = (TextRunner)runner;

        GenerationRequest request = new GenerationRequest
        {
            Prompt = prompt,
            MaxTokens = parameters.GetInt("max-tokens", 256),
            Sampling = BuildSampling(parameters),
        };

        List<int> streamed = new List<int>();
        int printedChars = 0;

        Stopwatch sw = Stopwatch.StartNew();
        GenerationResult result = text.Pipeline.Generate(request, tokenId =>
        {
            streamed.Add(tokenId);
            string decoded = text.Tokenizer.Decode(streamed);
            if (decoded.Length > printedChars)
            {
                progress.Token(decoded[printedChars..]);
                printedChars = decoded.Length;
            }
        });
        sw.Stop();

        double seconds = Math.Max(sw.Elapsed.TotalSeconds, 1e-6);
        int generated = Math.Max(1, result.TokenIds.Count);

        GeneratedArtifact artifact = new GeneratedArtifact { Kind = ArtifactKind.Text, Text = result.Text, Extension = "txt" };
        artifact.Meta["model"] = text.ModelId;
        artifact.Meta["prompt_tokens"] = result.PromptTokens.ToString(CultureInfo.InvariantCulture);
        artifact.Meta["generated_tokens"] = result.TokenIds.Count.ToString(CultureInfo.InvariantCulture);
        artifact.Meta["tokens_per_sec"] = (generated / seconds).ToString("F1", CultureInfo.InvariantCulture);
        artifact.Meta["stopped_on"] = result.StoppedOnStopToken ? "stop-token" : "max-tokens";
        return artifact;
    }

    private static TextRunner LoadGguf(ModelSpec spec, string path, IBackend backend, bool cuda, IProgressSink progress)
    {
        progress.Stage($"Loading GGUF {Path.GetFileName(path)} …");
        bool lowVram = Environment.GetEnvironmentVariable("HARTSY_LOWVRAM") == "1";
        // The CPU backend is F32-only; widen quantized projections at load for it. CUDA dequantizes on-device.
        GgufLanguageModel gguf = GgufLanguageModel.Load(path, lowVram, dequantizeToF32: !cuda);
        TransformerConfig cfg = gguf.Config;
        progress.Stage($"  arch={gguf.Architecture} hidden={cfg.HiddenSize} layers={cfg.NumLayers} vocab={cfg.VocabSize}");

        if (cuda)
        {
            progress.Stage("Preloading weights to GPU …");
            backend.PreloadWeights(gguf.Transformer.EnumerateWeights());
        }

        TextGenerationPipeline pipeline = new TextGenerationPipeline(gguf.Transformer, gguf.Tokenizer, backend, gguf.Template);
        string id = spec.Catalog?.Id ?? Path.GetFileNameWithoutExtension(path);
        return new TextRunner(id, pipeline, gguf.Tokenizer, ownedModel: gguf, ownedLoader: null);
    }

    private static TextRunner LoadSafetensors(ModelSpec spec, string dir, IBackend backend, bool cuda, IProgressSink progress)
    {
        if (Directory.GetFiles(dir, "*.safetensors").Length == 0)
            throw new FileNotFoundException($"No .safetensors weights found under {dir}.");

        string arch = (spec.Catalog?.Id ?? "qwen2").ToLowerInvariant();
        TransformerConfig cfg = arch switch
        {
            "qwen2" => TransformerConfig.Qwen2_5_0_5B,
            "qwen3" => TransformerConfig.Qwen3_0_6B,
            _ => throw new NotSupportedException(
                $"Safetensors text arch '{arch}' has no config preset yet — convert to GGUF, or select qwen2/qwen3."),
        };

        progress.Stage($"Loading {Path.GetFileName(dir)} …");
        SafeTensorsShardLoader loader = new SafeTensorsShardLoader();
        loader.LoadDirectory(dir);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();

        if (!cuda)
        {
            foreach (string key in weights.Keys.ToList())
            {
                if (weights[key].DType != DType.F32)
                    weights[key] = weights[key].CastTo(DType.F32);
            }
        }

        GenericTransformer model = new GenericTransformer(cfg);
        model.LoadWeights(weights, "model");

        string vocabPath = Path.Combine(dir, "vocab.json");
        ILlmTokenizer tokenizer = File.Exists(vocabPath)
            ? new Qwen2Tokenizer(vocabPath, Path.Combine(dir, "merges.txt"))
            : new Qwen2Tokenizer();

        if (cuda)
        {
            progress.Stage("Preloading weights to GPU …");
            backend.PreloadWeights(model.EnumerateWeights());
        }

        TextGenerationPipeline pipeline = new TextGenerationPipeline(model, tokenizer, backend, template: null);
        string id = spec.Catalog?.Id ?? Path.GetFileName(dir.TrimEnd(Path.DirectorySeparatorChar));
        return new TextRunner(id, pipeline, tokenizer, ownedModel: null, ownedLoader: loader);
    }

    private static SamplingOptions BuildSampling(ParamState parameters)
    {
        float temperature = parameters.GetFloat("temperature", 0.7f);
        if (temperature <= 0f)
            return SamplingOptions.GreedyPreset;

        int seed = parameters.GetInt("seed", -1);
        return new SamplingOptions
        {
            Temperature = temperature,
            TopP = parameters.GetFloat("top-p", 0.95f),
            Seed = seed < 0 ? 0UL : (ulong)seed,
        };
    }
}
