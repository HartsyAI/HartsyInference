using System.Diagnostics;
using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight regression harness for the Lens GPT-OSS NVFP4 text-encoder load path. The old
/// load eagerly dequantized the 20B-parameter MoE expert bank to F32 (~76 GB) and got the host process
/// OOM-killed on 64 GB boxes; the fixed path keeps the packed bank mmap-backed and streams one expert
/// at a time at forward time. This test loads the real ~13 GB ComfyUI checkpoint, runs TWO back-to-back
/// real encodes (native heap corruption from backend misuse historically surfaced on the second), and
/// asserts peak host RSS (<c>/proc/self/status VmHWM</c>) stays far below the danger zone.
/// <para>Backend: CPU by default; set <c>LENS_TE_BACKEND=cuda</c> to run the encode through a
/// <see cref="CudaBackend"/> (combine with <c>CUDA_VISIBLE_DEVICES</c> to pick the GPU) — this is the
/// SwarmUI production configuration that crashed with <c>malloc(): unsorted double linked list
/// corrupted</c> before the sequential-expert CUDA path landed.</para></summary>
public sealed class LensTeMemoryBoundedLoadTests
{
    private readonly ITestOutputHelper _output;

    public LensTeMemoryBoundedLoadTests(ITestOutputHelper output) => _output = output;

    /// <summary>Peak-RSS budget: well under the 62 GB host and the ~25 GB target from the incident
    /// review. Includes the mmap-resident packed weights (~13 GB, reclaimable file-backed pages) plus
    /// the F32 copies of the non-expert BF16 tensors (~5 GB).</summary>
    private const long PeakRssBudgetKb = 25L * 1024 * 1024;

    [Fact]
    [Trait("Category", "Integration")]
    public void RealNvfp4TextEncoder_LoadAndEncode_PeakRssBounded()
    {
        string tePath = Environment.GetEnvironmentVariable("LENS_TE_NVFP4_PATH")
            ?? "/home/hartsy/Desktop/Swarm/SwarmUI.not too old/Models/clip/gpt_oss_20b_nvfp4.safetensors";
        if (!File.Exists(tePath))
        {
            _output.WriteLine($"SKIPPED: NVFP4 text encoder not found at {tePath}");
            return;
        }

        using IBackend backend = CreateBackend(out string backendName);
        _output.WriteLine($"Backend: {backendName}");

        _output.WriteLine($"VmHWM before load: {ReadVmHwmKb() / 1048576.0:F2} GB");
        Stopwatch sw = Stopwatch.StartNew();

        using SafeTensorsLoader loader = new();
        loader.Load(tePath);
        Dictionary<string, Tensor> converted = LensCheckpointConverter.ConvertComfyTextEncoder(loader.GetAllTensors());
        using LensGptOssEncoder encoder = new();
        encoder.LoadWeights(converted);

        long afterLoadKb = ReadVmHwmKb();
        _output.WriteLine($"Loaded {converted.Count} tensors in {sw.Elapsed.TotalSeconds:F1}s; " +
                          $"VmHWM after load: {afterLoadKb / 1048576.0:F2} GB");
        Assert.True(afterLoadKb < PeakRssBudgetKb,
            $"Peak RSS after TE load is {afterLoadKb / 1048576.0:F2} GB — expected < {PeakRssBudgetKb / 1048576.0:F0} GB.");

        // TWO real encodes back-to-back. 112 synthetic-but-valid token ids (> the 97-token chat wrapper
        // the Lens front-end strips) exercise embedding, all 24 blocks including the streamed NVFP4 MoE,
        // and the 4-layer capture path over the real weights. The second pass catches state corruption
        // (stale GPU cache entries, use-after-free of reused tensors) that a single pass can miss.
        for (int pass = 0; pass < 2; pass++)
        {
            int[] tokenIds = new int[112];
            for (int i = 0; i < tokenIds.Length; i++) tokenIds[i] = 1000 + i * 13 + pass;

            sw.Restart();
            List<Tensor> features = encoder.EncodeForLens(backend, tokenIds);
            sw.Stop();

            long afterEncodeKb = ReadVmHwmKb();
            _output.WriteLine($"Encode pass {pass + 1} (112 tokens, 24 layers, {backendName}) took " +
                              $"{sw.Elapsed.TotalSeconds:F1}s; VmHWM: {afterEncodeKb / 1048576.0:F2} GB");
            try
            {
                Assert.Equal(4, features.Count);
                foreach (Tensor feature in features)
                {
                    Assert.Equal(new long[] { 1, 112 - LensGptOssEncoder.DefaultTextOffset, 2880 },
                        new[] { feature.Shape[0], feature.Shape[1], feature.Shape[2] });
                    AssertFiniteAndNonTrivial(feature);
                }
                Assert.True(afterEncodeKb < PeakRssBudgetKb,
                    $"Peak RSS after encode is {afterEncodeKb / 1048576.0:F2} GB — expected < {PeakRssBudgetKb / 1048576.0:F0} GB.");
            }
            finally
            {
                foreach (Tensor feature in features) feature.Dispose();
            }
        }
    }

    /// <summary>CPU backend by default; <c>LENS_TE_BACKEND=cuda</c> builds a <see cref="CudaBackend"/>
    /// on device 0 of the visible set with the test bin's PTX directory.</summary>
    private IBackend CreateBackend(out string backendName)
    {
        if (!string.Equals(Environment.GetEnvironmentVariable("LENS_TE_BACKEND"), "cuda", StringComparison.OrdinalIgnoreCase))
        {
            backendName = "CPU";
            return new CpuBackend();
        }

        string assemblyDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        backendName = $"CUDA (CUDA_VISIBLE_DEVICES={Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES") ?? "unset"})";
        return new CudaBackend(deviceOrdinal: 0, ptxDir: Directory.Exists(ptxDir) ? ptxDir : null);
    }

    private static unsafe void AssertFiniteAndNonTrivial(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long n = t.Shape.ElementCount;
        double sumAbs = 0;
        for (long i = 0; i < n; i++)
        {
            Assert.True(float.IsFinite(p[i]), $"non-finite value {p[i]} at flat index {i}");
            sumAbs += Math.Abs(p[i]);
        }
        Assert.True(sumAbs > 0, "captured hidden state is all zeros");
    }

    /// <summary>Reads the process peak resident set size (VmHWM) in KiB from /proc/self/status.</summary>
    private static long ReadVmHwmKb()
    {
        foreach (string line in File.ReadLines("/proc/self/status"))
        {
            if (!line.StartsWith("VmHWM:", StringComparison.Ordinal)) continue;
            // Format: "VmHWM:\t14829684 kB" — the separator run mixes tabs and spaces.
            string[] parts = line.Split([' ', '\t'], StringSplitOptions.RemoveEmptyEntries);
            return long.Parse(parts[1]);
        }
        throw new InvalidOperationException("VmHWM not found in /proc/self/status.");
    }
}
