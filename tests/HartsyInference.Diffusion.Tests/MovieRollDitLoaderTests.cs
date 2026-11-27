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
            output.WriteLine($"converter output {converted.Count} tensors for the top level and block 0");
            Assert.True(converted.Count > 0);
            // Pass-through tensors are the loader's mapped views and are released with the loader; only the converter's copies are disposed here.
            HashSet<Tensor> borrowed = new(subset.Values, ReferenceEqualityComparer.Instance);
            foreach (Tensor tensor in converted.Values)
            {
                if (!borrowed.Contains(tensor)) tensor.Dispose();
            }
        }
        finally
        {
            // The mapped views borrow the loader's memory: they are not disposed on their own.
            subset.Clear();
        }
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
