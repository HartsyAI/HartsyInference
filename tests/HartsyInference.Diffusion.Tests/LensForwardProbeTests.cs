using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Single deterministic Lens DiT forward over real weights with seeded synthetic inputs, dumping every
/// layer via <c>LENS_DEBUG_DIR</c>. Not an assertion test — it produces the layer dumps that before/after
/// implementations are diffed against (the GPU-residency refactor's parity probe). Env-gated on
/// <c>LENS_DIT_CHECKPOINT</c> + <c>LENS_PROBE_DUMP_DIR</c>.</summary>
public sealed class LensForwardProbeTests
{
    private readonly ITestOutputHelper _output;
    public LensForwardProbeTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void Forward_DumpLayers_Gpu()
    {
        string? ditPath = Environment.GetEnvironmentVariable("LENS_DIT_CHECKPOINT");
        string? dumpDir = Environment.GetEnvironmentVariable("LENS_PROBE_DUMP_DIR");
        if (ditPath is null || dumpDir is null || !File.Exists(ditPath))
        {
            _output.WriteLine("SKIPPED: set LENS_DIT_CHECKPOINT + LENS_PROBE_DUMP_DIR.");
            return;
        }
        string assemblyDir = Path.GetDirectoryName(typeof(LensForwardProbeTests).Assembly.Location)!;
        if (!Directory.Exists(Path.Combine(assemblyDir, "Ptx")) || !CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: PTX dir or CUDA driver unavailable.");
            return;
        }
        Environment.SetEnvironmentVariable("LENS_DEBUG_DIR", dumpDir);
        Directory.CreateDirectory(dumpDir);

        const int hPacked = 16, wPacked = 16, sTxt = 20;
        Stopwatch sw = Stopwatch.StartNew();
        using SafeTensorsLoader loader = new();
        loader.Load(ditPath);
        Dictionary<string, Tensor> ditWeights = LensCheckpointConverter.ConvertComfyDit(loader.GetAllTensors());
        LensConfig config = LensConfig.Turbo;
        using LensTransformer transformer = new(config);
        transformer.LoadWeights(ditWeights);
        _output.WriteLine($"Weights loaded in {sw.ElapsedMilliseconds}ms");

        using CudaBackend backend = new(deviceOrdinal: 0, Path.Combine(assemblyDir, "Ptx"));
        Tensor packed = SeedGenerator.CreateNoise(new TensorShape(1, hPacked * wPacked, config.InChannels), 42);
        Tensor[] encLayers = new Tensor[4];
        for (int i = 0; i < 4; i++)
            encLayers[i] = SeedGenerator.CreateNoise(new TensorShape(1, sTxt, config.EncoderHiddenDim), 43 + i);

        sw.Restart();
        Tensor output = transformer.Forward(backend, packed, encLayers, timestep: 0.7f, hPacked, wPacked);
        backend.Sync();
        _output.WriteLine($"Forward done in {sw.ElapsedMilliseconds}ms; output {output.Shape}");
        output.Dispose();
        packed.Dispose();
        for (int i = 0; i < 4; i++) encLayers[i].Dispose();
    }
}
