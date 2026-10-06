using System.Collections.Concurrent;
using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Dispatch;
using HartsyInference.LLM.Decision.Clef;
using HartsyInference.LLM.Ssm;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Services;

/// <summary>Cloudflare Clef typed decisions. A checkpoint is a Hugging Face directory (sharded backbone, <c>joint_head.safetensors</c>,
/// <c>tokenizer.json</c>); one pipeline is cached per directory.</summary>
public sealed class DecisionService : IDecisionService, IDisposable
{
    private readonly InferenceEngine _engine;
    private readonly ConcurrentDictionary<string, Loaded> _models = new(StringComparer.OrdinalIgnoreCase);

    internal DecisionService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<string> DecideAsync(ModelSpec spec, string requestJson, CancellationToken cancel = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(requestJson);
        string? path = spec.LocalPath;
        if (string.IsNullOrEmpty(path))
        {
            throw new HartsyInferenceException($"No checkpoint found for model '{spec.Requested}'. Pass the release directory via the model spec.");
        }
        string dir = Directory.Exists(path) ? path : Path.GetDirectoryName(path)!;
        Loaded model = _models.GetOrAdd(dir, Load);
        cancel.ThrowIfCancellationRequested();
        return Task.FromResult(model.Pipeline.Decide(_engine.Backend, requestJson));
    }

    /// <summary>Releases every cached pipeline and the checkpoint mappings behind it.</summary>
    public void Dispose()
    {
        foreach (Loaded model in _models.Values)
        {
            model.Dispose();
        }
        _models.Clear();
    }

    private static Loaded Load(string dir)
    {
        string head = Path.Combine(dir, "joint_head.safetensors");
        string tokenizerPath = Path.Combine(dir, "tokenizer.json");
        if (!File.Exists(head) || !File.Exists(tokenizerPath) || !File.Exists(Path.Combine(dir, "config.json")))
        {
            throw new HartsyInferenceException($"'{dir}' is not a Clef release (needs config.json, joint_head.safetensors and tokenizer.json).");
        }
        List<SafeTensorsLoader> loaders = [];
        try
        {
            Dictionary<string, Tensor> backbone = new(StringComparer.Ordinal);
            foreach (string shard in Directory.GetFiles(dir, "model-*.safetensors").Order(StringComparer.Ordinal))
            {
                SafeTensorsLoader loader = new();
                loader.Load(shard);
                loaders.Add(loader);
                foreach (string name in loader.Descriptors.Keys)
                {
                    if (name.StartsWith("model.language_model.", StringComparison.Ordinal) || name == "lm_head.weight")
                    {
                        backbone[name] = loader.GetTensor(name);
                    }
                }
            }
            SafeTensorsLoader headLoader = new();
            headLoader.Load(head);
            loaders.Add(headLoader);
            Dictionary<string, Tensor> headWeights = headLoader.Descriptors.Keys.ToDictionary(k => k, headLoader.GetTensor);
            using JsonDocument headJson = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "joint_head_config.json")));
            JsonElement hc = headJson.RootElement;
            ClefJointHeadConfig headConfig = new()
            {
                HiddenSize = hc.GetProperty("hidden_size").GetInt32(), Width = hc.GetProperty("width").GetInt32(),
                RoutingLayers = hc.GetProperty("routing_layers").GetInt32(), Layers = hc.GetProperty("layers").GetInt32(),
                Heads = hc.GetProperty("heads").GetInt32(), Feedforward = hc.GetProperty("feedforward").GetInt32(),
            };
            using FileStream tokenizerStream = File.OpenRead(tokenizerPath);
            GgufTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(tokenizerStream);
            Qwen35HfConfig config = Qwen35HfConfig.FromJson(File.ReadAllText(Path.Combine(dir, "config.json")));
            ClefDecisionPipeline pipeline = ClefDecisionPipeline.Create(backbone, headWeights, config, headConfig, tokenizer);
            return new Loaded(pipeline, loaders);
        }
        catch
        {
            foreach (SafeTensorsLoader l in loaders)
            {
                l.Dispose();
            }
            throw;
        }
    }

    private sealed class Loaded(ClefDecisionPipeline pipeline, List<SafeTensorsLoader> loaders) : IDisposable
    {
        public ClefDecisionPipeline Pipeline { get; } = pipeline;

        public void Dispose()
        {
            Pipeline.Dispose();
            foreach (SafeTensorsLoader l in loaders)
            {
                l.Dispose();
            }
        }
    }
}
