using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>Loads a SANA-WM DiT safetensors file under the upstream key names, such as <c>blocks.0.attn.qkv.weight</c>.</summary>
/// <remarks>Keys are not renamed to the diffusers layout. Dtype policy: BF16 for linear and convolution weights and for
/// <c>y_embedder.y_embedding</c>, which is kept as stored. F32 for biases, norms, GDN and gate parameters (<c>A_log</c>,
/// <c>dt_bias</c>, <c>beta_proj</c>, <c>gate_proj</c>, <c>recall_gate</c>), the adaLN path (<c>t_block</c>,
/// <c>t_embedder</c>), every <c>scale_shift_table</c> and <c>pos_embed</c>. Source F32 is rounded to BF16 with
/// round-to-nearest-even, as PyTorch's conversion does. Every tensor is copied into memory the returned
/// <see cref="SanaWmCheckpoint"/> owns, and the file mapping is released before <see cref="Load"/> returns.</remarks>
public static class SanaWmCheckpointConverter
{
    /// <summary>Loads and validates a SANA-WM DiT file. Every key must match the header-derived layout.</summary>
    /// <exception cref="HartsyInferenceException">The file is not safetensors, or its keys, shapes or dtypes do not match the layout.</exception>
    /// <param name="include">Optional filter on upstream key names. Every key is still validated against the header; only
    /// the tensors it accepts are copied, so a test can load one block without the whole 10.65 GB file.</param>
    public static SanaWmCheckpoint Load(string path, Func<string, bool>? include = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        SanaWmConfig config = SanaWmConfig.FromWeights(loader.Descriptors);
        IReadOnlyList<string> problems = ValidateKeys(config, loader.Descriptors);
        if (problems.Count > 0)
        {
            throw new HartsyInferenceException(
                $"SANA-WM checkpoint '{path}' does not match the layout its header implies "
                + $"({problems.Count} mismatches):\n  - " + string.Join("\n  - ", problems));
        }

        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal);
        try
        {
            foreach (SanaWmTensorSpec spec in ExpectedTensors(config))
            {
                if (include is null || include(spec.Name))
                    weights.Add(spec.Name, CopyAs(loader.GetTensor(spec.Name), spec.DType));
            }
        }
        catch
        {
            foreach (Tensor tensor in weights.Values)
                tensor.Dispose();
            throw;
        }
        return new SanaWmCheckpoint(config, weights);
    }

    /// <summary>Lists every difference between the header and the layout <paramref name="config"/> implies: missing keys,
    /// unexpected keys, shape mismatches and source dtypes other than F32 or BF16. An empty list means the file matches.</summary>
    public static IReadOnlyList<string> ValidateKeys(
        SanaWmConfig config,
        IReadOnlyDictionary<string, SafeTensorDescriptor> header)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(header);
        IReadOnlyList<SanaWmTensorSpec> expected = ExpectedTensors(config);
        HashSet<string> expectedNames = new(expected.Count, StringComparer.Ordinal);
        List<string> problems = [];
        foreach (SanaWmTensorSpec spec in expected)
        {
            expectedNames.Add(spec.Name);
            if (!header.TryGetValue(spec.Name, out SafeTensorDescriptor? actual))
            {
                problems.Add($"missing {spec.Name} ({spec.DType.Name} {SanaWmConfig.ShapeText(spec.Shape)})");
                continue;
            }
            if (actual.Shape != spec.Shape)
                problems.Add($"{spec.Name}: shape {SanaWmConfig.ShapeText(actual.Shape)}, "
                    + $"expected {SanaWmConfig.ShapeText(spec.Shape)}");
            if (actual.DType != DType.F32 && actual.DType != DType.BF16)
                problems.Add($"{spec.Name}: dtype {actual.DType.Name}, expected F32 or BF16");
        }
        foreach (string name in header.Keys.Where(key => !expectedNames.Contains(key)).Order(StringComparer.Ordinal))
        {
            SafeTensorDescriptor extra = header[name];
            problems.Add($"unexpected {name} ({extra.DType.Name} {SanaWmConfig.ShapeText(extra.Shape)})");
        }
        return problems;
    }

    /// <summary>Every tensor the DiT expects for <paramref name="config"/>, in checkpoint order, with the dtype it is stored as.</summary>
    public static IReadOnlyList<SanaWmTensorSpec> ExpectedTensors(SanaWmConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        long c = config.HiddenSize;
        long heads = config.NumHeads;
        long ffn = config.FfnHidden;
        long kernel = config.ConvKernelSize;
        long rows = config.ModulationRows;
        long outChannels = config.OutChannels;
        List<SanaWmTensorSpec> specs = [];

        Add(specs, "x_embedder.proj.weight", DType.BF16, c, config.InChannels, 1, 1, 1);
        Add(specs, "x_embedder.proj.bias", DType.F32, c);
        Add(specs, "t_embedder.mlp.0.weight", DType.F32, c, config.TimestepWidth);
        Add(specs, "t_embedder.mlp.0.bias", DType.F32, c);
        Add(specs, "t_embedder.mlp.2.weight", DType.F32, c, c);
        Add(specs, "t_embedder.mlp.2.bias", DType.F32, c);
        Add(specs, "t_block.1.weight", DType.F32, rows * c, c);
        Add(specs, "t_block.1.bias", DType.F32, rows * c);
        Add(specs, "y_embedder.y_proj.fc1.weight", DType.BF16, c, config.TextWidth);
        Add(specs, "y_embedder.y_proj.fc1.bias", DType.F32, c);
        Add(specs, "y_embedder.y_proj.fc2.weight", DType.BF16, c, c);
        Add(specs, "y_embedder.y_proj.fc2.bias", DType.F32, c);
        Add(specs, "y_embedder.y_embedding", DType.BF16, config.NullCaptionLength, config.TextWidth);
        Add(specs, "pos_embed", DType.F32, 1, config.PosEmbedTokens, c);
        Add(specs, "attention_y_norm.weight", DType.F32, c);
        if (config.HasPluckerBranch)
        {
            Add(specs, "plucker_embedder.proj.weight", DType.BF16, c, config.PluckerChannels, 1, 1, 1);
            Add(specs, "plucker_embedder.proj.bias", DType.F32, c);
        }
        if (config.HasRaymapEmbedder)
        {
            Add(specs, "raymap_embedder.proj.weight", DType.BF16, c, config.RaymapChannels, 1, 1, 1);
            Add(specs, "raymap_embedder.proj.bias", DType.F32, c);
        }
        Add(specs, "final_layer.linear.weight", DType.BF16, outChannels, c);
        Add(specs, "final_layer.linear.bias", DType.F32, outChannels);
        Add(specs, "final_layer.scale_shift_table", DType.F32, 2, c);

        for (int block = 0; block < config.Depth; block++)
        {
            string b = $"blocks.{block}.";
            Add(specs, b + "attn.qkv.weight", DType.BF16, 3 * c, c);
            Add(specs, b + "attn.proj.weight", DType.BF16, c, c);
            Add(specs, b + "attn.proj.bias", DType.F32, c);
            Add(specs, b + "attn.output_gate.weight", DType.BF16, c, c);
            Add(specs, b + "attn.output_gate.bias", DType.F32, c);
            Add(specs, b + "attn.q_norm.weight", DType.F32, c);
            Add(specs, b + "attn.k_norm.weight", DType.F32, c);
            Add(specs, b + "attn.A_log", DType.F32, heads);
            Add(specs, b + "attn.dt_bias", DType.F32, heads);
            Add(specs, b + "attn.beta_proj.weight", DType.F32, heads, c);
            Add(specs, b + "attn.beta_proj.bias", DType.F32, heads);
            Add(specs, b + "attn.gate_proj.weight", DType.F32, heads, c);
            Add(specs, b + "attn.gate_proj.bias", DType.F32, heads);
            Add(specs, b + "attn.recall_gate", DType.F32, 1);
            if (!config.IsSoftmaxBlock(block))
            {
                Add(specs, b + "attn.conv_k.weight", DType.BF16, c, 1, kernel);
                if (config.HasCameraBranch)
                    Add(specs, b + "attn.conv_k_cam.weight", DType.BF16, c, 1, kernel);
            }
            if (config.HasCameraBranch)
            {
                AddLinear(specs, b + "attn.q_proj_cam", c, c);
                AddLinear(specs, b + "attn.k_proj_cam", c, c);
                AddLinear(specs, b + "attn.v_proj_cam", c, c);
                AddLinear(specs, b + "attn.out_proj_cam", c, c);
                Add(specs, b + "attn.q_norm_cam.weight", DType.F32, c);
                Add(specs, b + "attn.k_norm_cam.weight", DType.F32, c);
            }
            if (config.HasPluckerBranch)
                AddLinear(specs, b + "plucker_proj", c, c);
            Add(specs, b + "scale_shift_table", DType.F32, rows, c);
            AddLinear(specs, b + "cross_attn.q_linear", c, c);
            AddLinear(specs, b + "cross_attn.kv_linear", 2 * c, c);
            AddLinear(specs, b + "cross_attn.proj", c, c);
            Add(specs, b + "cross_attn.q_norm.weight", DType.F32, c);
            Add(specs, b + "cross_attn.k_norm.weight", DType.F32, c);
            Add(specs, b + "mlp.inverted_conv.conv.weight", DType.BF16, 2 * ffn, c, 1, 1);
            Add(specs, b + "mlp.inverted_conv.conv.bias", DType.F32, 2 * ffn);
            Add(specs, b + "mlp.depth_conv.conv.weight", DType.BF16, 2 * ffn, 1, 3, 3);
            Add(specs, b + "mlp.depth_conv.conv.bias", DType.F32, 2 * ffn);
            Add(specs, b + "mlp.point_conv.conv.weight", DType.BF16, c, ffn, 1, 1);
            Add(specs, b + "mlp.t_conv.weight", DType.BF16, c, c, 3, 1);
        }
        return specs;
    }

    /// <summary>Adds a bias-carrying linear layer: a BF16 weight and an F32 bias.</summary>
    private static void AddLinear(List<SanaWmTensorSpec> specs, string prefix, long outFeatures, long inFeatures)
    {
        Add(specs, prefix + ".weight", DType.BF16, outFeatures, inFeatures);
        Add(specs, prefix + ".bias", DType.F32, outFeatures);
    }

    private static void Add(List<SanaWmTensorSpec> specs, string name, DType dtype, params long[] dims) =>
        specs.Add(new SanaWmTensorSpec(name, dtype, new TensorShape(dims)));

    /// <summary>Returns an owned copy of <paramref name="source"/> stored as <paramref name="target"/>. Same-dtype copies are
    /// byte copies; F32 to BF16 uses the shared round-to-nearest-even conversion.</summary>
    private static unsafe Tensor CopyAs(Tensor source, DType target)
    {
        long count = source.Shape.ElementCount;
        Tensor owned = new(source.Shape, target);
        try
        {
            if (source.DType == target)
            {
                long bytes = target.ComputeByteCount(count);
                Buffer.MemoryCopy(source.DataPointer, owned.DataPointer, bytes, bytes);
            }
            else
            {
                SafeTensorsMerger.Convert(
                    (byte*)source.DataPointer, source.DType, (byte*)owned.DataPointer, target, checked((int)count));
            }
            return owned;
        }
        catch
        {
            owned.Dispose();
            throw;
        }
    }
}
