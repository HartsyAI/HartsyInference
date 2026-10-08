using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.BlockScale;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Builds a <see cref="DeepSeekV41VisionModel"/> from an opened DeepSeek-V4.1 checkpoint, widening the <c>vision.*</c>, <c>aligner.*</c> and <c>image_*</c> tensors to F32.</summary>
/// <remarks>The official checkpoint stores all 266 of these as BF16 without quantization companions (about 1 GB in its first two shards), so widening is exact. The returned model owns copies, not views, and does not need
/// the checkpoint afterwards.</remarks>
public static class DeepSeekV41VisionLoader
{
    /// <summary>Loads the vision tower, aligner and image-span embeddings of <paramref name="checkpoint"/>.</summary>
    /// <exception cref="HartsyInferenceException">The config declares no vision tower, a tensor is missing, or its shape differs from the config.</exception>
    public static DeepSeekV41VisionModel Load(IBackend backend, DeepSeekV41Checkpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(checkpoint);
        DeepSeekV41VisionConfig config = checkpoint.Config.Vision
            ?? throw new HartsyInferenceException($"The config at {checkpoint.Info.ConfigPath} declares no vision tower.");
        config.Validate();
        long dim = config.HiddenSize, outDim = checkpoint.Config.HiddenSize;

        Reader read = new(checkpoint);
        try
        {
            DeepSeekV41VisionBlockWeights[] blocks = new DeepSeekV41VisionBlockWeights[config.NumLayers];
            for (int i = 0; i < blocks.Length; i++) blocks[i] = ReadBlock(read, config, i);
            DeepSeekV41VisionWeights towerWeights = new(read.Matrix("vision.patch_embed.proj.weight", dim, config.PatchInputDim),
                read.Vector("vision.patch_embed.proj.bias", dim), blocks, read.Vector("vision.norm.weight", dim));
            DeepSeekV41AlignerWeights alignerWeights = new(read.Matrix("aligner.w1.weight", outDim, config.AlignerInputDim),
                read.Vector("aligner.w1.bias", outDim), read.Matrix("aligner.w2.weight", outDim, outDim), read.Vector("aligner.w2.bias", outDim));
            float[] imageStart = read.Values("image_start", outDim);
            float[] imageEnd = read.Values("image_end", outDim);
            float[] imageNewline = read.Values("image_newline", outDim);
            DeepSeekV41VisionModel model = new(new DeepSeekV41VisionTower(backend, config, towerWeights),
                new DeepSeekV41Aligner(backend, config, (int)outDim, alignerWeights), imageStart, imageEnd, imageNewline);
            read.Release();
            return model;
        }
        finally
        {
            // Tensor.Dispose is idempotent, so tensors a half-built tower or aligner already holds are safe to free here
            read.DisposeUnreleased();
        }
    }

    private static DeepSeekV41VisionBlockWeights ReadBlock(Reader read, DeepSeekV41VisionConfig config, int index)
    {
        long dim = config.HiddenSize, inter = config.IntermediateSize;
        string b = $"vision.blocks.{index}.";
        return new DeepSeekV41VisionBlockWeights(
            read.Vector(b + "norm1.weight", dim), read.Matrix(b + "attn.wqkv.weight", 3 * dim, dim), read.Vector(b + "attn.wqkv.bias", 3 * dim),
            read.Matrix(b + "attn.wo.weight", dim, dim), read.Vector(b + "attn.wo.bias", dim), read.Vector(b + "norm2.weight", dim),
            read.Matrix(b + "mlp.w1.weight", 2 * inter, dim), read.Matrix(b + "mlp.w2.weight", dim, inter));
    }

    // Reads canonical keys as F32 tensors, remembering them so a failed load frees what it had read.
    private sealed class Reader(DeepSeekV41Checkpoint checkpoint)
    {
        private readonly List<Tensor> _tensors = [];
        private bool _released;

        public Tensor Matrix(string key, long rows, long cols)
        {
            Require(key);
            float[] values = DeepSeekV41ExpertLoader.ReadMatrix(key, checkpoint.GetWeight(key), checkpoint.GetQuant(key), rows, cols);
            return Keep(DeepSeekV41HostMath.Tensor(values, rows, cols));
        }

        public Tensor Vector(string key, long length) => Keep(DeepSeekV41HostMath.Tensor(Values(key, length), length));

        public float[] Values(string key, long length)
        {
            Require(key);
            float[] values = WeightDequantizer.ToF32(checkpoint.GetWeight(key), checkpoint.GetQuant(key));
            if (values.Length != length) throw new HartsyInferenceException($"{key} holds {values.Length} values, expected {length}.");
            return values;
        }

        // the model owns the tensors from here on
        public void Release() => _released = true;

        public void DisposeUnreleased()
        {
            if (_released) return;
            foreach (Tensor tensor in _tensors) tensor.Dispose();
        }

        private Tensor Keep(Tensor tensor)
        {
            _tensors.Add(tensor);
            return tensor;
        }

        private void Require(string key)
        {
            if (!checkpoint.HasWeight(key))
                throw new HartsyInferenceException($"The checkpoint at {checkpoint.Info.Root} has no '{key}', which the vision tower needs.");
        }
    }
}
