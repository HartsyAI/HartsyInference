using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Ernie5;
using HartsyInference.LLM.Maxwell;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Hy4Flash;
using HartsyInference.LLM.Glm5Next;
using HartsyInference.LLM.Kimi;
using HartsyInference.LLM.Models.Lfm3;
using HartsyInference.LLM.Models.Spark;
using HartsyInference.LLM.Ssm;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Services;

/// <summary>Opens a Hugging Face safetensors directory for the Text path; DeepSeek-V4.1, ERNIE-5.5 and Kimi K3.1 (host reference models), stateful families (LFM3, Spark-X) and MiniMax-M3 are recognised.</summary>
internal static class HfTextDirectoryLoader
{
    /// <summary>The <c>model_type</c> of a Kimi K3.1 checkpoint.</summary>
    internal const string KimiK31ModelType = "kimi_k31";

    /// <summary>Longest sequence (prompt plus generation) a loaded V4.1 or ERNIE model accepts. It sizes the rope tables and every sequence state; the checkpoint's own limit is far higher, but a host reference run does not need it.</summary>
    internal const int MaxSequenceTokens = 16384;

    /// <summary>Longest sequence (prompt plus generation) a loaded stateful-family model accepts; it sizes the KV cache of its attention layers.</summary>
    internal const int StatefulMaxSequenceTokens = 8192;

    /// <summary>The options the Text path loads with; the RAM guard and the memory profile size against the same ones.</summary>
    internal static DeepSeekV41LoadOptions LoadOptions { get; } = new(MaxSequenceTokens);

    /// <summary>The ERNIE-5.5 options the Text path loads with on <paramref name="device"/>. A GPU keeps every layer built and preloads each layer's
    /// per-token weights while it has room. The CPU reads, dequantizes and drops each layer on every forward, since resident layers there would only
    /// add host memory.</summary>
    internal static Ernie5LoadOptions Ernie5OptionsFor(DeviceKind device) => device.IsGpu ? Ernie5Resident : Ernie5Streamed;

    /// <summary>The Maxwell options the Text path loads with on <paramref name="device"/>: resident layers on a GPU, read-and-drop layers on the CPU,
    /// as ERNIE-5.5 does.</summary>
    internal static MaxwellLoadOptions MaxwellOptionsFor(DeviceKind device) => new() { MaxSequenceTokens = MaxSequenceTokens, ResidentWeights = device.IsGpu };

    private static readonly Ernie5LoadOptions Ernie5Resident = new() { MaxSequenceTokens = MaxSequenceTokens, ResidentWeights = true };
    private static readonly Ernie5LoadOptions Ernie5Streamed = new() { MaxSequenceTokens = MaxSequenceTokens, ResidentWeights = false };

    /// <summary>Whether <paramref name="info"/> is a host reference model (DeepSeek-V4.1, ERNIE-5.5 or Kimi K3.1) that <see cref="LoadHostReference"/> serves.</summary>
    internal static bool IsHostReferenceFamily(HfCheckpointInfo info) =>
        string.Equals(info.ModelType, Ernie5Config.ModelType, StringComparison.Ordinal)
        || string.Equals(info.ModelType, MaxwellConfig.ModelType, StringComparison.Ordinal)
        || string.Equals(info.ModelType, DeepSeekV41Config.ModelType, StringComparison.Ordinal)
        || IsHy4Flash(info)
        || IsKimiK31(info)
        || IsGlm5Next(info);

    /// <summary>Whether <paramref name="info"/> is a glm5_next checkpoint (Sage S2.1 Flash, GLM-5.3-Flash). It runs on the backend of the requested device.</summary>
    internal static bool IsGlm5Next(HfCheckpointInfo info) => string.Equals(info.ModelType, Glm5NextTextModel.ModelType, StringComparison.Ordinal);

    /// <summary>Whether <paramref name="info"/> is a Hy4 Flash checkpoint (<c>hy_v4_flash</c>, the base and instruct builds alike).</summary>
    internal static bool IsHy4Flash(HfCheckpointInfo info) => string.Equals(info.ModelType, Hy4FlashConfig.ModelType, StringComparison.Ordinal);

    /// <summary>Whether <paramref name="info"/> is a Kimi K3.1 checkpoint. Its backend follows the requested device rather than the CPU.</summary>
    internal static bool IsKimiK31(HfCheckpointInfo info) => string.Equals(info.ModelType, KimiK31ModelType, StringComparison.Ordinal);

    /// <summary>Loads a host reference directory onto <paramref name="backend"/>. DeepSeek-V4.1 keeps its weights on the host, so the caller passes a CPU
    /// backend; ERNIE-5.5 and Kimi K3.1 run on the backend they are given.</summary>
    /// <exception cref="HartsyInferenceException">The directory's model_type is not supported, it has no usable tokenizer.json, or the checkpoint is invalid.</exception>
    /// <param name="ernie5">ERNIE-5.5 load options in place of <see cref="Ernie5OptionsFor"/>, such as a layer split over several devices.</param>
    internal static IHostReferenceTextModel LoadHostReference(HfCheckpointInfo info, IBackend backend, Ernie5LoadOptions? ernie5 = null)
    {
        RequireSupported(info);
        if (string.Equals(info.ModelType, Ernie5Config.ModelType, StringComparison.Ordinal))
            return Ernie5TextModel.Load(backend, info.Root, ernie5 ?? Ernie5OptionsFor(backend.Device));
        if (string.Equals(info.ModelType, MaxwellConfig.ModelType, StringComparison.Ordinal))
            return MaxwellTextModel.Load(backend, info.Root, MaxwellOptionsFor(backend.Device));
        if (IsKimiK31(info))
            return KimiK31TextModel.Load(backend, info.Root, MaxSequenceTokens);
        if (IsHy4Flash(info))
            return Hy4FlashTextModel.Load(backend, info.Root, MaxSequenceTokens);
        if (IsGlm5Next(info))
            return Glm5NextTextModel.Load(backend, info.Root, MaxSequenceTokens);
        return DeepSeekV41TextModel.Load(backend, info.Root, LoadOptions);
    }

    /// <summary>Loads a Kimi K3.1 directory sharded over <paramref name="rankBackends"/>, one backend per rank; <paramref name="comm"/> is the collective
    /// over them and is required with two or more ranks.</summary>
    internal static IHostReferenceTextModel LoadKimiK31Sharded(HfCheckpointInfo info, IReadOnlyList<IBackend> rankBackends, ICollectiveComm? comm)
    {
        RequireSupported(info);
        if (!IsKimiK31(info)) throw new ArgumentException($"'{info.ModelType}' is not a Kimi K3.1 checkpoint.", nameof(info));
        return KimiK31TextModel.Load(rankBackends, comm, info.Root, MaxSequenceTokens);
    }

    /// <summary>The decoder config of a Kimi K3.1 checkpoint directory, read before any weight is loaded.</summary>
    internal static KimiK31Config ReadKimiK31Config(HfCheckpointInfo info) =>
        KimiK31HfConfig.Parse(System.IO.File.ReadAllText(System.IO.Path.Combine(info.Root, "config.json")));

    /// <summary>Whether <paramref name="info"/> is a family that carries its own state (LFM3, Spark-X) rather than DeepSeek-V4.1.</summary>
    internal static bool IsStatefulFamily(HfCheckpointInfo info) => SsmLanguageModel.IsHuggingFaceFamily(info.ModelType);

    /// <summary>Loads a stateful-family directory with its tokenizer and chat template.</summary>
    internal static SsmLanguageModel LoadStateful(HfCheckpointInfo info) =>
        SsmLanguageModel.LoadHuggingFace(info.Root, info.ModelType, StatefulMaxSequenceTokens);

    /// <summary>Loads the MiniMax-M3 directory's text decoder, tokenizer and chat template as the model object the transformer slot serves. Matrices are 4-bit blocks on a backend that reads them,
    /// else F32 (the CPU backend widens quantized weights anyway); <paramref name="memoryBudgetBytes"/> of 0 means no limit.</summary>
    internal static GgufLanguageModel LoadMiniMaxM3(HfCheckpointInfo info, IBackend backend, long memoryBudgetBytes)
    {
        MiniMaxM3HfLoader loader = MiniMaxM3HfLoader.Load(info.Root, backend.Capabilities.SupportsQuantized ? DType.Q4_0 : DType.F32, memoryBudgetBytes);
        try
        {
            GenericTransformer transformer = new(loader.Config);
            transformer.LoadWeights(loader.Weights, "model");
            (ILlmTokenizer tokenizer, IChatTemplate template) = HfChatAssets.Load(info.Root, [], MiniMaxM3Bos, MiniMaxM3Eos);
            return GgufLanguageModel.FromWeights(loader.Config, transformer, tokenizer, template, loader);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    private const string MiniMaxM3Bos = "]~b]";
    private const string MiniMaxM3Eos = "[e~[";

    /// <summary>Refuses a directory the Text path cannot run for a reason visible without opening a weight (an unsupported model_type, no tokenizer.json), so the caller can do it before it unloads anything.</summary>
    /// <exception cref="HartsyInferenceException">The model_type is not supported or the tokenizer is missing.</exception>
    internal static void RequireSupported(HfCheckpointInfo info)
    {
        if (string.Equals(info.ModelType, Ernie5Config.ModelType, StringComparison.Ordinal))
            Ernie5Config.RequireRunnable(info.Root, info.ConfigPath);
        if (string.Equals(info.ModelType, MaxwellConfig.ModelType, StringComparison.Ordinal))
        {
            MaxwellConfig.Parse(File.ReadAllText(info.ConfigPath));
            // Refused here, before TextService unloads the model it replaces: the Maxwell Text path needs the template.
            if (!File.Exists(Path.Combine(info.Root, "chat_template.jinja")))
                throw new HartsyInferenceException($"'{info.Root}' has no chat_template.jinja, which the Maxwell Text path needs.");
        }
        if (IsKimiK31(info))
            RequireKimiK31Runnable(info);
        if (IsHy4Flash(info))
            Hy4FlashConfig.Load(info.ConfigPath);
        if (IsGlm5Next(info))
        {
            Glm5NextTextModel.ReadConfig(info.ConfigPath);
            if (!File.Exists(Path.Combine(info.Root, "chat_template.jinja")))
                throw new HartsyInferenceException($"'{info.Root}' has no chat_template.jinja, which the Sage S2.1 Flash Text path needs.");
        }
        bool stateful = IsStatefulFamily(info);
        bool miniMax = string.Equals(info.ModelType, MiniMaxM3HfConfigReader.ModelType, StringComparison.Ordinal);
        if (!stateful && !miniMax && !IsHostReferenceFamily(info))
        {
            throw new HartsyInferenceException(
                $"'{info.Root}' is a Hugging Face checkpoint with model_type '{info.ModelType}', which the Text path cannot load; "
                + "pass a .gguf file instead.");
        }
        if (miniMax && !File.Exists(Path.Combine(info.Root, "chat_template.jinja")))
            throw new HartsyInferenceException($"'{info.Root}' has no chat_template.jinja, which the MiniMax-M3 Text path needs.");
        if (stateful)
        {
            bool single = string.Equals(info.ModelType, Lfm3Config.ModelType, StringComparison.Ordinal);
            bool hasWeights = single ? File.Exists(Path.Combine(info.Root, "model.safetensors")) : Directory.EnumerateFiles(info.Root, "*.safetensors").Any();
            if (!hasWeights)
                throw new HartsyInferenceException($"'{info.Root}' has no {(single ? "model.safetensors" : "*.safetensors file")}, which the {info.ModelType} Text path needs.");
            if (!File.Exists(Path.Combine(info.Root, "chat_template.jinja")))
                throw new HartsyInferenceException($"'{info.Root}' has no chat_template.jinja, which the {info.ModelType} Text path needs.");
        }
        // A Kimi K3.1 directory may carry tiktoken.model instead; RequireKimiK31Runnable has already checked for one of the two.
        if (!IsKimiK31(info) && !File.Exists(Path.Combine(info.Root, "tokenizer.json")))
            throw new HartsyInferenceException($"'{info.Root}' has no tokenizer.json, which the Text path needs to build its prompt.");
    }

    /// <summary>Refuses a Kimi K3.1 directory the Text path cannot read before any weight is opened. The tokenizer is tokenizer.json or, when that is absent,
    /// tiktoken.model; the template is chat_template.jinja or, on a tiktoken.model checkpoint, the built-in XTML template.</summary>
    /// <exception cref="HartsyInferenceException">The shard index is missing, the directory has neither tokenizer file, or it has tokenizer.json without a chat template.</exception>
    private static void RequireKimiK31Runnable(HfCheckpointInfo info)
    {
        if (info.IndexPath is null)
            throw new HartsyInferenceException($"'{info.Root}' has no model.safetensors.index.json, which the Kimi K3.1 Text path needs to find its shards.");
        bool hasTokenizerJson = File.Exists(Path.Combine(info.Root, "tokenizer.json"));
        bool hasTiktoken = File.Exists(Path.Combine(info.Root, KimiK31Tokenizer.TiktokenFile));
        if (!hasTokenizerJson && !hasTiktoken)
            throw new HartsyInferenceException($"'{info.Root}' has no tokenizer.json or tiktoken.model, which the Kimi K3.1 Text path needs to build its prompt.");
        if (hasTokenizerJson && !File.Exists(Path.Combine(info.Root, "chat_template.jinja")))
            throw new HartsyInferenceException($"'{info.Root}' has no chat_template.jinja, which the Kimi K3.1 Text path needs.");
    }
}
