using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.LLM.Multimodal;

namespace HartsyInference.TextGen.Cli;

/// <summary>Reference-parity harness for the mllama (Llama-3.2-Vision) vision tower. Loads the mmproj, feeds the
/// fixed pixel input from <c>dump_mllama_vision_ref.py</c>, and dumps each stage so the C# tower can be diffed
/// against the numpy reference to corr 1.0 (the standard bar for every VLM tower in this engine).</summary>
public static class MllamaVisRunner
{
    public static unsafe int Run(string backendName)
    {
        string? mmproj = Environment.GetEnvironmentVariable("HARTSY_MMPROJ");
        string? pixels = Environment.GetEnvironmentVariable("HARTSY_MLLAMA_PIXELS");
        if (mmproj is null || !File.Exists(mmproj)) { Console.Error.WriteLine("Set HARTSY_MMPROJ to the mllama mmproj .gguf."); return 1; }
        if (pixels is null || !File.Exists(pixels)) { Console.Error.WriteLine("Set HARTSY_MLLAMA_PIXELS to a raw [3,560,560] f32."); return 1; }

        using IBackend backend = backendName == "cpu"
            ? new CpuBackend()
            : new CudaBackend(deviceOrdinal: 0, ptxDir: Path.Combine(AppContext.BaseDirectory, "Ptx"));

        Console.WriteLine($"Loading mmproj {Path.GetFileName(mmproj)} ...");
        using MllamaVisionEncoder enc = MllamaVisionEncoder.Load(mmproj);
        Console.WriteLine($"  hidden={enc.Hidden} local={enc.NumLocalLayers} global={enc.NumGlobalLayers} heads={enc.NumHeads} seq={enc.SeqLen} projDim={enc.ProjectionDim}");

        int s = enc.ImageSize;
        byte[] raw = File.ReadAllBytes(pixels);
        if (raw.Length != 3 * s * s * 4) { Console.Error.WriteLine($"pixels file is {raw.Length} bytes, expected {3 * s * s * 4} ([3,{s},{s}] f32)."); return 1; }
        using Tensor px = new(new TensorShape(1, 3, s, s), DType.F32);
        fixed (byte* rp = raw) Buffer.MemoryCopy(rp, (void*)px.DataPointer, raw.Length, raw.Length);

        using Tensor embeds = enc.Encode(backend, px);
        backend.Sync();
        float* e = (float*)embeds.DataPointer;
        Console.WriteLine($"embeds shape [{embeds.Shape[1]},{embeds.Shape[2]}] first6 = [{e[0]:F4}, {e[1]:F4}, {e[2]:F4}, {e[3]:F4}, {e[4]:F4}, {e[5]:F4}]");
        Console.WriteLine($"Stage dumps → {Environment.GetEnvironmentVariable("HARTSY_MLLAMA_DUMP") ?? "(set HARTSY_MLLAMA_DUMP to diff)"}");
        return 0;
    }
}
