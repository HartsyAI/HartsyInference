using System.Runtime.InteropServices;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The engine's CPU F32 forward of the released MovieRoll DiT with one block loaded at a time. The shared weights are loaded once; each block is
/// read from the file into owned buffers just before its forward (the DiT's <c>BeforeBlockForward</c> seam), converted and loaded, and the previous
/// block's buffers are released once the next block has loaded. Reading owned buffers, not mapped views, keeps the working set to one block.
/// Environment: <c>MOVIEROLL_DIT_PATH</c> the released file, <c>MOVIEROLL_INPUTS</c> the saved inputs, <c>MOVIEROLL_VELOCITY_OUT</c> the output (raw
/// little-endian floats), <c>MOVIEROLL_BLOCKS</c> the blocks to run (default 54; a smaller number is a memory dry run whose output is not compared),
/// and <c>MOVIEROLL_BLOCK_LOG</c> a file that receives one timestamped line per block load.</summary>
[Trait("Category", "Integration")]
public sealed class MovieRollBlockStreamedForwardTests(ITestOutputHelper output)
{
    [RealWeightsFact("MOVIEROLL_DIT_PATH", "MOVIEROLL_INPUTS")]
    public void BlockAtATimeCpuForward_OnTheUpstreamInputs_WritesTheVelocity()
    {
        string dit = Environment.GetEnvironmentVariable("MOVIEROLL_DIT_PATH")!;
        string inputsPath = Environment.GetEnvironmentVariable("MOVIEROLL_INPUTS")!;
        string outPath = Environment.GetEnvironmentVariable("MOVIEROLL_VELOCITY_OUT") ?? "engine_velocity.bin";
        string? logPath = Environment.GetEnvironmentVariable("MOVIEROLL_BLOCK_LOG");
        int blocks = int.TryParse(Environment.GetEnvironmentVariable("MOVIEROLL_BLOCKS"), out int b) ? b : 54;

        HunyuanVideo15Config config = HunyuanVideo15Config.Full with { NumBlocks = blocks };
        using SafeTensorsLoader weights = new();
        weights.Load(dit);
        using SafeTensorsLoader inputs = new();
        inputs.Load(inputsPath);
        using CpuBackend backend = new();
        using FileStream file = File.OpenRead(dit);

        void Log(string line)
        {
            output.WriteLine(line);
            if (!string.IsNullOrEmpty(logPath)) File.AppendAllText(logPath, $"{DateTime.Now:HH:mm:ss} {line}{Environment.NewLine}");
        }

        // Owned copies of the named tensors, read from the file: each buffer belongs to the caller and is released with Dispose.
        Dictionary<string, Tensor> ReadOwned(IEnumerable<string> names)
        {
            Dictionary<string, Tensor> read = new(StringComparer.Ordinal);
            foreach (string name in names)
            {
                SafeTensorDescriptor descriptor = weights.Descriptors[name];
                Tensor tensor = new(descriptor.Shape, descriptor.DType);
                byte[] buffer = new byte[descriptor.ByteLength];
                file.Seek(descriptor.DataOffset, SeekOrigin.Begin);
                file.ReadExactly(buffer);
                buffer.AsSpan().CopyTo(MemoryMarshal.AsBytes(tensor.AsSpan<float>()));
                read[name] = tensor;
            }
            return read;
        }

        static IEnumerable<string> BlockNames(string prefix, IEnumerable<string> all) =>
            all.Where(name => name.StartsWith(prefix, StringComparison.Ordinal));

        IEnumerable<string> allNames = weights.Descriptors.Keys;
        IEnumerable<string> topNames = allNames.Where(name => !name.StartsWith("double_blocks.", StringComparison.Ordinal)).ToList();

        // Shared weights: the top level and block 0 are read; the converter runs over them, the shared outputs stay alive for the DiT, and the
        // block-0 outputs of this conversion are released (the hook loads block 0 again from its own read).
        Dictionary<string, Tensor> sharedInputs = ReadOwned(topNames.Concat(BlockNames("double_blocks.0.", allNames)));
        Dictionary<string, Tensor> sharedRenamed = new(sharedInputs, StringComparer.Ordinal);
        Dictionary<string, Tensor> sharedOut = HunyuanVideo15CheckpointConverter.Convert(sharedRenamed, numBlocks: 1, hidden: config.HiddenSize);
        HashSet<Tensor> sharedKeep = new(ReferenceEqualityComparer.Instance);
        foreach (KeyValuePair<string, Tensor> pair in sharedOut)
        {
            if (!pair.Key.StartsWith("transformer_blocks.0.", StringComparison.Ordinal)) sharedKeep.Add(pair.Value);
        }
        HashSet<Tensor> sharedInputSet = new(sharedInputs.Values, ReferenceEqualityComparer.Instance);
        using HunyuanVideo15Dit dit15 = new(config);
        Dictionary<string, Tensor> sharedOnly = sharedOut
            .Where(pair => !pair.Key.StartsWith("transformer_blocks.0.", StringComparison.Ordinal))
            .ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal);
        dit15.LoadSharedWeights(sharedOnly);
        foreach (KeyValuePair<string, Tensor> pair in sharedOut)
        {
            // A block-0 output that is also a shared input is released with the inputs at the end, not here.
            if (pair.Key.StartsWith("transformer_blocks.0.", StringComparison.Ordinal) && !sharedInputSet.Contains(pair.Value)) pair.Value.Dispose();
        }
        // Inputs the shared weights use are kept (the refiner and the embedders read them); block-0 inputs are released.
        List<Tensor> previous = [];
        Log($"shared weights loaded: {sharedOnly.Count} tensors");

        dit15.BeforeBlockForward = index =>
        {
            Dictionary<string, Tensor> blockRead = ReadOwned(BlockNames($"double_blocks.{index}.", allNames));
            Dictionary<string, Tensor> converterInput = new(StringComparer.Ordinal);
            foreach (KeyValuePair<string, Tensor> pair in sharedInputs)
            {
                if (!pair.Key.StartsWith("double_blocks.0.", StringComparison.Ordinal)) converterInput[pair.Key] = pair.Value;
            }
            foreach (KeyValuePair<string, Tensor> pair in blockRead)
            {
                converterInput["double_blocks.0." + pair.Key[$"double_blocks.{index}.".Length..]] = pair.Value;
            }
            Dictionary<string, Tensor> converted = HunyuanVideo15CheckpointConverter.Convert(converterInput, numBlocks: 1, hidden: config.HiddenSize);
            Dictionary<string, Tensor> blockWeights = new(StringComparer.Ordinal);
            HashSet<Tensor> owned = new(ReferenceEqualityComparer.Instance);
            foreach (Tensor tensor in blockRead.Values) owned.Add(tensor);
            foreach (KeyValuePair<string, Tensor> pair in converted)
            {
                if (sharedInputSet.Contains(pair.Value) || sharedKeep.Contains(pair.Value)) continue;
                if (pair.Key.StartsWith("transformer_blocks.0.", StringComparison.Ordinal))
                {
                    blockWeights["transformer_blocks." + index + pair.Key["transformer_blocks.0".Length..]] = pair.Value;
                    owned.Add(pair.Value);
                }
                else
                {
                    owned.Add(pair.Value); // top-level copies from this conversion are not used
                }
            }
            dit15.LoadBlockWeights(index, blockWeights);
            foreach (Tensor old in previous) old.Dispose();
            previous = owned.Where(tensor => !sharedInputSet.Contains(tensor) && !sharedKeep.Contains(tensor)).ToList();
            Log($"loaded block {index}: {blockWeights.Count} tensors, released {previous.Count} buffers of the last block");
        };

        Tensor qwen = inputs.GetTensor("qwen");
        Tensor byt5 = inputs.GetTensor("byt5");
        Tensor noise = inputs.GetTensor("noise");
        float[] frameTimesteps = [1000f, 555.56f, 769.23f, 882.35f, 952.38f];
        DateTime start = DateTime.UtcNow;
        using Tensor velocity = dit15.ForwardPerFrame(backend, noise, frameTimesteps, 1000f, qwen, byt5, conditioning: null);
        Log($"forward {(DateTime.UtcNow - start).TotalSeconds:F1} s over {blocks} blocks; velocity shape {velocity.Shape}");

        ReadOnlySpan<float> values = velocity.AsReadOnlySpan<float>();
        using (FileStream stream = File.Create(outPath))
        using (BinaryWriter writer = new(stream))
        {
            foreach (float v in values) writer.Write(v);
        }
        foreach (Tensor old in previous) old.Dispose();
        foreach (Tensor tensor in sharedInputs.Values) tensor.Dispose();
    }
}
