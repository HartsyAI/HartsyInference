using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.CheckpointConverters.Utils;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.ModelHandler.CheckpointConverters;

/// <summary>Loads Matrix-Game 3.0 DiT checkpoints (<c>Skywork/Matrix-Game-3.0</c>,
/// <c>base_model/diffusion_pytorch_model.safetensors</c>) for <c>MatrixGame3Transformer</c>. The core Wan2.2 keys are
/// in original Wan naming and reuse <see cref="WanVideoCheckpointConverter.MapKey"/> (same rename table); the
/// Matrix-Game additions — <c>blocks.{i}.action_model.*</c> (or <c>action_module.*</c>, normalized) and the Plücker
/// projection (<c>patch_embedding_wancamctrl.*</c>, <c>c2ws_hidden_states_layer1/2.*</c>) — pass through in original
/// naming and are routed BEFORE the Wan renames (whose <c>cross_attn</c>/<c>modulation</c> rules would corrupt them).
///
/// <para>The DMD-distilled bundle (<c>base_distilled_model/</c>, ~2× the base size) carries student + critic/EMA
/// copies; <see cref="Convert"/> auto-slices a recognized student prefix and drops the rest (prefix set is
/// validation-gated until a key dump). <see cref="InferShape"/> resolves the dim/layers/ffn/heads contradiction
/// between the repo config (5120/40/40) and the Wan-AI diffusers config (3072/24/30) from the weights themselves.
/// VAE: the same <c>wan22_vae.safetensors</c> as Lance/Wan via <see cref="LanceCheckpointConverter.LoadVae"/>;
/// MG-LightVAE pruned decoders are a follow-up. umT5-XXL ships separately.</para></summary>
public sealed class MatrixGame3CheckpointConverter
{
    private static readonly string[] _studentPrefixes = ["student.", "generator."];
    private static readonly string[] _dropPrefixes = ["critic.", "fake_score.", "ema.", "discriminator."];

    /// <summary>Result bucket: one dictionary holding the diffusers-renamed Wan core plus the original-named
    /// Matrix-Game additions, exactly what <c>MatrixGame3Transformer.LoadWeights</c> consumes.</summary>
    public sealed class ConvertedWeights
    {
        public required Dictionary<string, Tensor> Transformer { get; init; }

        /// <summary>Block indices that carry ActionModule weights.</summary>
        public required int[] ActionBlocks { get; init; }
    }

    /// <summary>Inferred DiT shape, read from the weights (the authoritative answer to Open Question 2).</summary>
    public readonly record struct InferredShape(int Dim, int NumLayers, int FfnDim, int NumHeads, int InChannels);

    /// <summary>Pure key mapping: returns the destination key, or <c>null</c> to drop. Matrix-Game-specific keys
    /// (action modules, Plücker projection) bypass the Wan rename table; <c>action_module</c> is normalized to
    /// <c>action_model</c>.</summary>
    public static string? MapKey(string key, bool fromOriginalNaming)
    {
        foreach (string drop in _dropPrefixes)
            if (key.StartsWith(drop, StringComparison.Ordinal))
                return null;
        foreach (string student in _studentPrefixes)
            if (key.StartsWith(student, StringComparison.Ordinal))
                key = key[student.Length..];

        if (key.Contains(".action_model.", StringComparison.Ordinal))
            return key;
        if (key.Contains(".action_module.", StringComparison.Ordinal))
            return key.Replace(".action_module.", ".action_model.", StringComparison.Ordinal);
        if (key.StartsWith("patch_embedding_wancamctrl", StringComparison.Ordinal)
            || key.StartsWith("c2ws_hidden_states_layer", StringComparison.Ordinal)
            || key.StartsWith("plucker", StringComparison.Ordinal))
            return key;

        return WanVideoCheckpointConverter.MapKey(key, fromOriginalNaming);
    }

    /// <summary>Converts a flat weight dictionary: student-slice extraction (distilled bundles), Matrix-Game key
    /// routing, Wan original→diffusers renames, FP8 scale folding.</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights)
    {
        allWeights = CheckpointConvertUtils.ApplyFp8ScaledDequant(allWeights);

        // If a student prefix exists, only that slice is the inference model.
        string? studentPrefix = null;
        foreach (string candidate in _studentPrefixes)
        {
            foreach (string key in allWeights.Keys)
                if (key.StartsWith(candidate, StringComparison.Ordinal)) { studentPrefix = candidate; break; }
            if (studentPrefix is not null) break;
        }

        Dictionary<string, Tensor> source = allWeights;
        if (studentPrefix is not null)
        {
            source = new Dictionary<string, Tensor>(allWeights.Count);
            foreach (KeyValuePair<string, Tensor> kvp in allWeights)
                if (kvp.Key.StartsWith(studentPrefix, StringComparison.Ordinal))
                    source[kvp.Key] = kvp.Value;
        }

        bool original = WanVideoCheckpointConverter.IsOriginalNaming(source.Keys);
        Dictionary<string, Tensor> transformer = new(source.Count);
        SortedSet<int> actionBlocks = new();
        foreach (KeyValuePair<string, Tensor> kvp in source)
        {
            string? mapped = MapKey(kvp.Key, original);
            if (mapped is null) continue;
            transformer[mapped] = kvp.Value;
            int actionIdx = mapped.IndexOf(".action_model.", StringComparison.Ordinal);
            if (actionIdx > 0 && mapped.StartsWith("blocks.", StringComparison.Ordinal)
                && int.TryParse(mapped.AsSpan("blocks.".Length, actionIdx - "blocks.".Length), out int blockIdx))
                actionBlocks.Add(blockIdx);
        }
        return new ConvertedWeights { Transformer = transformer, ActionBlocks = [.. actionBlocks] };
    }

    /// <summary>Reads the real DiT shape from converted weights: dim from <c>patch_embedding.weight</c>, layers from
    /// the max block index, ffn from <c>blocks.0.ffn.net.0.proj.weight</c>, heads = dim / 128 (Wan2.2 head_dim).</summary>
    public static InferredShape InferShape(IReadOnlyDictionary<string, Tensor> transformer)
    {
        Tensor patch = transformer["patch_embedding.weight"];
        int dim = (int)patch.Shape[0];
        int inChannels = (int)patch.Shape[1];
        int layers = 0;
        foreach (string key in transformer.Keys)
        {
            if (!key.StartsWith("blocks.", StringComparison.Ordinal)) continue;
            int dot = key.IndexOf('.', "blocks.".Length);
            if (dot > 0 && int.TryParse(key.AsSpan("blocks.".Length, dot - "blocks.".Length), out int idx))
                layers = Math.Max(layers, idx + 1);
        }
        int ffn = (int)transformer["blocks.0.ffn.net.0.proj.weight"].Shape[0];
        return new InferredShape(dim, layers, ffn, dim / 128, inChannels);
    }

    /// <summary>Loads a single safetensors file and converts. The caller owns the loader.</summary>
    public static (ConvertedWeights Weights, SafeTensorsLoader Loader) LoadAndConvert(string checkpointPath)
    {
        if (!File.Exists(checkpointPath))
            throw new FileNotFoundException($"Matrix-Game 3 checkpoint not found: {checkpointPath}");
        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        ConvertedWeights converted = Convert(loader.GetAllTensors());
        return (converted, loader);
    }

    /// <summary>Loads a model folder (e.g. <c>base_model/</c> with sharded safetensors) and converts.</summary>
    public static (ConvertedWeights Weights, List<SafeTensorsLoader> Loaders) LoadFolder(string modelDir)
    {
        if (!Directory.Exists(modelDir))
            throw new DirectoryNotFoundException($"Matrix-Game 3 model dir not found: {modelDir}");
        string[] shards = Directory.GetFiles(modelDir, "*.safetensors");
        if (shards.Length == 0)
            throw new FileNotFoundException($"No safetensors shards found in: {modelDir}");
        Array.Sort(shards, StringComparer.Ordinal);

        List<SafeTensorsLoader> loaders = new(shards.Length);
        Dictionary<string, Tensor> merged = new(4096);
        try
        {
            foreach (string shard in shards)
            {
                SafeTensorsLoader loader = new();
                loader.Load(shard);
                loaders.Add(loader);
                foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
                    merged[kvp.Key] = kvp.Value;
            }
            return (Convert(merged), loaders);
        }
        catch
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
            throw;
        }
    }
}
