using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The engine's CPU F32 forward of the released MovieRoll DiT with one block loaded at a time: the shared weights are loaded once, and each
/// block is converted and loaded just before its forward (the DiT's <c>BeforeBlockForward</c> seam), with the previous block's owned copies released
/// after the next one loads. Environment: <c>MOVIEROLL_DIT_PATH</c> names the released file, <c>MOVIEROLL_INPUTS</c> the saved inputs,
/// <c>MOVIEROLL_VELOCITY_OUT</c> the output file (raw little-endian floats), and <c>MOVIEROLL_BLOCKS</c> the number of blocks to run (default 54;
/// a smaller number is a memory dry run whose output is not compared).</summary>
[Trait("Category", "Integration")]
public sealed class MovieRollBlockStreamedForwardTests(ITestOutputHelper output)
{
    [RealWeightsFact("MOVIEROLL_DIT_PATH", "MOVIEROLL_INPUTS")]
    public void BlockAtATimeCpuForward_OnTheUpstreamInputs_WritesTheVelocity()
    {
        string dit = Environment.GetEnvironmentVariable("MOVIEROLL_DIT_PATH")!;
        string inputsPath = Environment.GetEnvironmentVariable("MOVIEROLL_INPUTS")!;
        string outPath = Environment.GetEnvironmentVariable("MOVIEROLL_VELOCITY_OUT") ?? "engine_velocity.bin";
        int blocks = int.TryParse(Environment.GetEnvironmentVariable("MOVIEROLL_BLOCKS"), out int b) ? b : 54;

        HunyuanVideo15Config config = HunyuanVideo15Config.Full with { NumBlocks = blocks };
        using SafeTensorsLoader weights = new();
        weights.Load(dit);
        using SafeTensorsLoader inputs = new();
        inputs.Load(inputsPath);
        using CpuBackend backend = new();

        Dictionary<string, Tensor> source = weights.GetAllTensors();
        HashSet<Tensor> mapped = new(source.Values, ReferenceEqualityComparer.Instance);

        // Shared weights: the converter over the top level and block 0; the block-0 outputs are not kept.
        Dictionary<string, Tensor> sharedAndBlock0 = HunyuanVideo15CheckpointConverter.Convert(
            source.Where(p => !p.Key.StartsWith("double_blocks.", StringComparison.Ordinal) || p.Key.StartsWith("double_blocks.0.", StringComparison.Ordinal))
                .ToDictionary(p => p.Key, p => p.Value), numBlocks: 1, hidden: config.HiddenSize);
        using HunyuanVideo15Dit dit15 = new(config);
        dit15.LoadSharedWeights(sharedAndBlock0);
        // The block-0 outputs of this conversion are not used (block 0 is loaded by the hook); the shared outputs stay alive for the DiT.
        foreach (KeyValuePair<string, Tensor> pair in sharedAndBlock0)
        {
            if (pair.Key.StartsWith("transformer_blocks.0.", StringComparison.Ordinal) && !mapped.Contains(pair.Value)) pair.Value.Dispose();
        }

        List<Tensor> previous = [];
        dit15.BeforeBlockForward = index =>
        {
            Dictionary<string, Tensor> one = new(StringComparer.Ordinal);
            string prefix = $"double_blocks.{index}.";
            foreach (KeyValuePair<string, Tensor> pair in source)
            {
                if (pair.Key.StartsWith(prefix, StringComparison.Ordinal)) one[pair.Key] = pair.Value;
                else if (!pair.Key.StartsWith("double_blocks.", StringComparison.Ordinal)) one[pair.Key] = pair.Value;
            }
            // Re-key block <index> as block 0 for the converter, which consumes block 0.
            Dictionary<string, Tensor> renamed = new(StringComparer.Ordinal);
            foreach (KeyValuePair<string, Tensor> pair in one)
            {
                string key = pair.Key.StartsWith(prefix, StringComparison.Ordinal) ? "double_blocks.0." + pair.Key[prefix.Length..] : pair.Key;
                renamed[key] = pair.Value;
            }
            Dictionary<string, Tensor> converted = HunyuanVideo15CheckpointConverter.Convert(renamed, numBlocks: 1, hidden: config.HiddenSize);
            Dictionary<string, Tensor> blockWeights = new(StringComparer.Ordinal);
            List<Tensor> owned = [];
            foreach (KeyValuePair<string, Tensor> pair in converted)
            {
                if (pair.Key.StartsWith("transformer_blocks.0.", StringComparison.Ordinal))
                {
                    blockWeights["transformer_blocks." + index + pair.Key["transformer_blocks.0".Length..]] = pair.Value;
                    if (!mapped.Contains(pair.Value)) owned.Add(pair.Value);
                }
                else if (!mapped.Contains(pair.Value))
                {
                    pair.Value.Dispose(); // top-level copies of this conversion are not used: the shared weights were loaded once
                }
            }
            dit15.LoadBlockWeights(index, blockWeights);
            // The new block references its own tensors now; the previous block's owned copies can go.
            foreach (Tensor old in previous) old.Dispose();
            previous = owned;
            output.WriteLine($"loaded block {index}: {blockWeights.Count} tensors, owned copies {owned.Count}");
        };

        Tensor qwen = inputs.GetTensor("qwen");
        Tensor byt5 = inputs.GetTensor("byt5");
        Tensor noise = inputs.GetTensor("noise");
        float[] frameTimesteps = [1000f, 555.56f, 769.23f, 882.35f, 952.38f];
        DateTime start = DateTime.UtcNow;
        using Tensor velocity = dit15.ForwardPerFrame(backend, noise, frameTimesteps, 1000f, qwen, byt5, conditioning: null);
        output.WriteLine($"forward {(DateTime.UtcNow - start).TotalSeconds:F1} s over {blocks} blocks; velocity shape {velocity.Shape}");

        ReadOnlySpan<float> values = velocity.AsReadOnlySpan<float>();
        using (FileStream stream = File.Create(outPath))
        using (BinaryWriter writer = new(stream))
        {
            foreach (float v in values) writer.Write(v);
        }
        foreach (Tensor old in previous) old.Dispose();
    }

}
