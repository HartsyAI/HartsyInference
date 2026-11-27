using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Shared;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Diagnostic: the released MovieRoll DiT's forward on one backend, block by block. Each block's output image tokens are written to
/// <c>MOVIEROLL_TAP_OUT</c> as <c>block_NN.bin</c> (raw little-endian floats), so the CPU and CUDA runs can be compared per block.
/// <c>MOVIEROLL_TAP_BACKEND</c> is <c>cpu</c> or <c>cuda</c>; the CUDA run is the exact lane (full-F32 GEMMs, DiT F16 activations off).
/// Environment: <c>MOVIEROLL_DIT_PATH</c>, <c>MOVIEROLL_INPUTS</c>. Diagnostic only; asserts nothing beyond the run completing.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class MovieRollBlockTapTests(ITestOutputHelper output)
{
    [Fact]
    public void BlockTap_WritesEachBlockOutput()
    {
        string? dit = Environment.GetEnvironmentVariable("MOVIEROLL_DIT_PATH");
        string? inputsPath = Environment.GetEnvironmentVariable("MOVIEROLL_INPUTS");
        string? outDir = Environment.GetEnvironmentVariable("MOVIEROLL_TAP_OUT");
        string backendName = Environment.GetEnvironmentVariable("MOVIEROLL_TAP_BACKEND") ?? "cpu";
        if (string.IsNullOrEmpty(dit) || string.IsNullOrEmpty(inputsPath) || string.IsNullOrEmpty(outDir))
        {
            output.WriteLine("SKIPPED: MOVIEROLL_DIT_PATH, MOVIEROLL_INPUTS or MOVIEROLL_TAP_OUT is unset.");
            return;
        }
        Directory.CreateDirectory(outDir);
        bool cuda = backendName == "cuda";

        HunyuanVideo15Config config = HunyuanVideo15Config.Full;
        using SafeTensorsLoader weights = new();
        weights.Load(dit);
        using SafeTensorsLoader inputs = new();
        inputs.Load(inputsPath);

        if (cuda) KnobStore.Set(EngineKnobs.DitF16, false);
        bool attentionF32 = Environment.GetEnvironmentVariable("MOVIEROLL_TAP_ATTENTION_F32") == "1";
        using IBackend backend = cuda ? GpuDevices.OpenCuda() : new CpuBackend();
        if (cuda) backend.HighPrecisionGemm = true;

        Dictionary<string, Tensor> source = weights.GetAllTensors();
        // Block weights are read into owned buffers (not mapped views), so each block's memory is released after the next block loads.
        using FileStream file = File.OpenRead(dit);
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
                buffer.AsSpan().CopyTo(System.Runtime.InteropServices.MemoryMarshal.AsBytes(tensor.AsSpan<float>()));
                read[name] = tensor;
            }
            return read;
        }

        Dictionary<string, Tensor> sharedInputs = ReadOwned(source.Keys.Where(k => !k.StartsWith("double_blocks.", StringComparison.Ordinal)
            || k.StartsWith("double_blocks.0.", StringComparison.Ordinal)).ToList());
        Dictionary<string, Tensor> sharedOut = HunyuanVideo15CheckpointConverter.Convert(sharedInputs, numBlocks: 1, hidden: config.HiddenSize);
        using HunyuanVideo15Dit dit15 = new(config);
        dit15.AllowF16Attention = !attentionF32;
        output.WriteLine($"attention F16 narrowing allowed: {!attentionF32}");
        dit15.LoadSharedWeights(sharedOut.Where(p => !p.Key.StartsWith("transformer_blocks.", StringComparison.Ordinal))
            .ToDictionary(p => p.Key, p => p.Value, StringComparer.Ordinal));

        List<Tensor> previous = [];
        Action<int> loadBlock = index =>
        {
            string prefix = $"double_blocks.{index}.";
            Dictionary<string, Tensor> blockRead = ReadOwned(source.Keys.Where(k => k.StartsWith(prefix, StringComparison.Ordinal)).ToList());
            Dictionary<string, Tensor> one = new(StringComparer.Ordinal);
            foreach (KeyValuePair<string, Tensor> pair in blockRead) one["double_blocks.0." + pair.Key[prefix.Length..]] = pair.Value;
            foreach (KeyValuePair<string, Tensor> pair in source)
            {
                if (!pair.Key.StartsWith("double_blocks.", StringComparison.Ordinal)) one[pair.Key] = pair.Value;
            }
            Dictionary<string, Tensor> converted = HunyuanVideo15CheckpointConverter.Convert(one, numBlocks: 1, hidden: config.HiddenSize);
            Dictionary<string, Tensor> blockWeights = new(StringComparer.Ordinal);
            List<Tensor> owned = [];
            HashSet<Tensor> mapped = new(source.Values, ReferenceEqualityComparer.Instance);
            owned.AddRange(blockRead.Values);
            foreach (KeyValuePair<string, Tensor> pair in converted)
            {
                if (pair.Key.StartsWith("transformer_blocks.0.", StringComparison.Ordinal))
                {
                    blockWeights["transformer_blocks." + index + pair.Key["transformer_blocks.0".Length..]] = pair.Value;
                    if (!mapped.Contains(pair.Value) && !blockRead.ContainsValue(pair.Value)) owned.Add(pair.Value);
                }
                else if (!mapped.Contains(pair.Value) && !blockRead.ContainsValue(pair.Value)) pair.Value.Dispose();
            }
            dit15.LoadBlockWeights(index, blockWeights);
            foreach (Tensor old in previous) old.Dispose();
            previous = owned;
        };
        dit15.BeforeBlockForward = loadBlock;

        BlockStreamingScope? stream = null;
        if (cuda)
        {
            stream = BlockStreamingScope.Open(new BlockStreamingOptions { Backend = backend, Denoiser = dit15, ModelName = "MovieRoll", HeadroomBytes = 6L << 30 });
            Action<int>? scopeHook = dit15.BeforeBlockForward;
            dit15.BeforeBlockForward = index => { loadBlock(index); scopeHook?.Invoke(index); };
        }
        try
        {
            dit15.BlockOutputTap = (index, image) =>
            {
                string path = Path.Combine(outDir, $"block_{index:D2}.bin");
                using FileStream stream2 = File.Create(path);
                using BinaryWriter writer = new(stream2);
                foreach (float v in image.AsReadOnlySpan<float>()) writer.Write(v);
            };
            Tensor qwen = inputs.GetTensor("qwen");
            Tensor byt5 = inputs.GetTensor("byt5");
            Tensor noise = inputs.GetTensor("noise");
            float[] frameTimesteps = [1000f, 555.56f, 769.23f, 882.35f, 952.38f];
            using Tensor velocity = dit15.ForwardPerFrame(backend, noise, frameTimesteps, 1000f, qwen, byt5, conditioning: null);
            stream?.EndStep();
            output.WriteLine($"block tap on {backendName}: {config.NumBlocks} blocks, velocity shape {velocity.Shape}");
        }
        finally
        {
            stream?.Dispose();
            foreach (Tensor old in previous) old.Dispose();
            KnobStore.ResetOverrides();
        }
    }
}
