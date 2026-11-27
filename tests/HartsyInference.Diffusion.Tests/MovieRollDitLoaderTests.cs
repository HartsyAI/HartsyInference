using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The released MovieRoll DiT file (<c>yunyangge/MovieRoll</c> <c>distilled/t2v/model.safetensors</c>, 33,291,800,480 bytes, sha256
/// 410694c3e24f6c5d3a53485fbebdfe8e9cc18165f103623eee21f6ff3b91b169), checked against the engine's own converter and the MovieRoll DiT
/// config. <c>MOVIEROLL_DIT_PATH</c> names the file; without it the tests are reported as skipped.
/// The header is read through the mapped loader, so no weight data is read except block 0, which the converter copies. Block 0 is converted
/// with the top-level tensors, and every other block must carry the same tensor names and shapes; any tensor the converter does not consume
/// fails the conversion. The file is FP32 throughout, which the test records.</summary>
[Trait("Category", "Integration")]
public sealed class MovieRollDitLoaderTests(ITestOutputHelper output)
{
    private const string PathVariable = "MOVIEROLL_DIT_PATH";

    [RealWeightsFact(PathVariable)]
    public void Header_IsFp32AndEveryBlockMatchesBlockZero()
    {
        string path = Environment.GetEnvironmentVariable(PathVariable)!;
        HunyuanVideo15Config config = HunyuanVideo15Config.Full;
        using SafeTensorsLoader loader = new();
        loader.Load(path);

        Dictionary<string, SafeTensorDescriptor> blockZero = [];
        Dictionary<string, SafeTensorDescriptor> topLevel = [];
        int[] seenBlocks = new int[config.NumBlocks];
        foreach (SafeTensorDescriptor descriptor in loader.Descriptors.Values)
        {
            Assert.Equal(DType.F32, descriptor.DType);
            (int? block, string suffix) = Split(descriptor.Name);
            if (block is null)
            {
                topLevel[descriptor.Name] = descriptor;
                continue;
            }
            Assert.InRange(block.Value, 0, config.NumBlocks - 1);
            seenBlocks[block.Value]++;
            if (block.Value == 0) blockZero[suffix] = descriptor;
        }
        output.WriteLine($"tensors {loader.Descriptors.Count}: top-level {topLevel.Count}, block 0 {blockZero.Count}, per block {string.Join(",", seenBlocks.Distinct())}");

        for (int block = 1; block < config.NumBlocks; block++)
        {
            Assert.Equal(blockZero.Count, seenBlocks[block]);
        }
        foreach (SafeTensorDescriptor descriptor in loader.Descriptors.Values)
        {
            (int? block, string suffix) = Split(descriptor.Name);
            if (block is null or 0) continue;
            Assert.True(blockZero.TryGetValue(suffix, out SafeTensorDescriptor? reference), $"{descriptor.Name} has no counterpart in block 0.");
            Assert.Equal(reference!.Shape, descriptor.Shape);
        }
    }

    [RealWeightsFact(PathVariable)]
    public void Converter_AcceptsTopLevelAndBlockZeroOfTheReleasedFile()
    {
        string path = Environment.GetEnvironmentVariable(PathVariable)!;
        HunyuanVideo15Config config = HunyuanVideo15Config.Full;
        using SafeTensorsLoader loader = new();
        loader.Load(path);

        // Every tensor except blocks 1..N-1: the top level and block 0. The converter throws when any of them is not consumed.
        Dictionary<string, Tensor> subset = [];
        foreach (string name in loader.Descriptors.Keys)
        {
            (int? block, _) = Split(name);
            if (block is null or 0) subset[name] = loader.GetTensor(name);
        }
        try
        {
            Dictionary<string, Tensor> converted = HunyuanVideo15CheckpointConverter.Convert(subset, numBlocks: 1, hidden: config.HiddenSize);
            try
            {
                // The engine's names and shapes for the top level and block 0, from the config: a missing or extra name, or a wrong shape, fails.
                Dictionary<string, long[]> expected = ExpectedConverted(config);
                List<string> problems = [];
                foreach (string name in expected.Keys.Where(name => !converted.ContainsKey(name))) problems.Add($"missing {name}");
                foreach (string name in converted.Keys.Where(name => !expected.ContainsKey(name))) problems.Add($"unexpected {name}");
                foreach ((string name, long[] dims) in expected)
                {
                    if (converted.TryGetValue(name, out Tensor? tensor) && tensor.Shape != new TensorShape(dims))
                        problems.Add($"{name} is {tensor.Shape}, expected [{string.Join(", ", dims)}]");
                }
                output.WriteLine($"converter output {converted.Count} tensors, expected {expected.Count}; problems {problems.Count}");
                Assert.True(problems.Count == 0, string.Join("; ", problems));

                // Load the one-block weights into a one-block DiT: the loader fails on any missing or misshapen weight it needs.
                using HunyuanVideo15Dit dit = new(config with { NumBlocks = 1 });
                dit.LoadWeights(converted);
                Assert.NotEmpty(dit.EnumerateWeights());
            }
            finally
            {
                // Pass-through tensors are the loader's mapped views and are released with the loader; only the converter's copies are disposed here.
                HashSet<Tensor> borrowed = new(subset.Values, ReferenceEqualityComparer.Instance);
                foreach (Tensor tensor in converted.Values)
                {
                    if (!borrowed.Contains(tensor)) tensor.Dispose();
                }
            }
        }
        finally
        {
            // The mapped views borrow the loader's memory: they are not disposed on their own.
            subset.Clear();
        }
    }

    /// <summary>The engine's converted names and shapes for the top level and block 0, from the config. The refiner has two blocks in the released
    /// file, and the timestep embedding is 256 wide.</summary>
    private static Dictionary<string, long[]> ExpectedConverted(HunyuanVideo15Config config)
    {
        long h = config.HiddenSize, mlp = config.MlpDim, text = config.TextDim, byt5 = config.ByT5Dim, latent = config.LatentChannels;
        long headDim = h / config.NumHeads;
        long imageIn = 2 * latent + 1;
        const long TimestepWidth = 256;
        Dictionary<string, long[]> e = new(StringComparer.Ordinal);
        void Add(string name, params long[] dims) => e[name] = dims;
        void Linear(string name, long outDim, long inDim) { Add($"{name}.weight", outDim, inDim); Add($"{name}.bias", outDim); }
        void Norm(string name, long dim) { Add($"{name}.weight", dim); Add($"{name}.bias", dim); }

        Norm("byt5_in.layernorm", byt5);
        Linear("byt5_in.fc1", h, byt5);
        Linear("byt5_in.fc2", h, h);
        Linear("byt5_in.fc3", h, h);
        Add("cond_type_embedding.weight", 3, h);
        Linear("final_layer.adaLN_modulation.1", 2 * h, h);
        Linear("final_layer.linear", latent, h);
        Linear("img_in.proj", h, imageIn);
        Linear("time_in.mlp.0", h, TimestepWidth);
        Linear("time_in.mlp.2", h, h);
        Linear("txt_in.c_embedder.in_layer", h, text);
        Linear("txt_in.c_embedder.out_layer", h, h);
        Linear("txt_in.input_embedder", h, text);
        Linear("txt_in.t_embedder.in_layer", h, TimestepWidth);
        Linear("txt_in.t_embedder.out_layer", h, h);
        for (int i = 0; i < 2; i++)
        {
            string p = $"txt_in.individual_token_refiner.blocks.{i}";
            Linear($"{p}.adaLN_modulation.1", 2 * h, h);
            Linear($"{p}.mlp.0", mlp, h);
            Linear($"{p}.mlp.2", h, mlp);
            Norm($"{p}.norm1", h);
            Norm($"{p}.norm2", h);
            Linear($"{p}.self_attn.proj", h, h);
            Linear($"{p}.self_attn.qkv", 3 * h, h);
        }
        const string block = "transformer_blocks.0";
        foreach (string proj in new[] { "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_add_out", "attn.to_out.0" })
            Linear($"{block}.{proj}", h, h);
        Add($"{block}.attn.norm_added_q.weight", headDim);
        Add($"{block}.attn.norm_added_k.weight", headDim);
        Add($"{block}.attn.norm_q.weight", headDim);
        Add($"{block}.attn.norm_k.weight", headDim);
        foreach (string ff in new[] { "ff", "ff_context" })
        {
            Linear($"{block}.{ff}.net.0.proj", mlp, h);
            Linear($"{block}.{ff}.net.2", h, mlp);
        }
        Linear($"{block}.norm1.linear", 6 * h, h);
        Linear($"{block}.norm1_context.linear", 6 * h, h);
        return e;
    }

    /// <summary>Splits <c>double_blocks.N.suffix</c> into its block and suffix; any other name has no block.</summary>
    private static (int? Block, string Suffix) Split(string name)
    {
        const string prefix = "double_blocks.";
        if (!name.StartsWith(prefix, StringComparison.Ordinal)) return (null, name);
        int dot = name.IndexOf('.', prefix.Length);
        if (dot < 0 || !int.TryParse(name.AsSpan(prefix.Length, dot - prefix.Length), out int block)) return (null, name);
        return (block, name[(dot + 1)..]);
    }
}
