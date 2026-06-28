using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Multimodal;

namespace HartsyInference.TextGen.Cli;

/// <summary>End-to-end Llama-3.2-Vision (mllama) generation. HARTSY_MODEL_DIR = the text .gguf, HARTSY_MMPROJ =
/// the mmproj .gguf. Feeds a synthetic colored shape (HARTSY_VLM_COLOR = r,g,b; HARTSY_VLM_PATTERN = circle|square)
/// so the answer is checkable, and runs the cross-attention generator.</summary>
public static class MllamaRunner
{
    // OpenAI CLIP normalization (mllama image processor).
    private static readonly float[] Mean = [0.48145466f, 0.4578275f, 0.40821073f];
    private static readonly float[] Std = [0.26862954f, 0.26130258f, 0.27577711f];

    public static unsafe int Run(string backendName, int genTokens, string question)
    {
        string? textPath = Environment.GetEnvironmentVariable("HARTSY_MODEL_DIR");
        string? mmproj = Environment.GetEnvironmentVariable("HARTSY_MMPROJ");
        if (textPath is null || !File.Exists(textPath)) { Console.Error.WriteLine("Set HARTSY_MODEL_DIR to the mllama text .gguf."); return 1; }
        if (mmproj is null || !File.Exists(mmproj)) { Console.Error.WriteLine("Set HARTSY_MMPROJ to the mllama mmproj .gguf."); return 1; }

        using IBackend backend = backendName == "cpu"
            ? new CpuBackend()
            : new CudaBackend(deviceOrdinal: 0, ptxDir: Path.Combine(AppContext.BaseDirectory, "Ptx"));
        CudaBackend? cuda = backend as CudaBackend;
        bool lowVram = Environment.GetEnvironmentVariable("HARTSY_LOWVRAM") == "1";

        Console.WriteLine($"Loading text {Path.GetFileName(textPath)} ...");
        using GgufLanguageModel text = GgufLanguageModel.Load(textPath, lowVram, dequantizeToF32: cuda is null);
        Console.WriteLine($"  arch={text.Architecture} hidden={text.Config.HiddenSize} layers={text.Config.NumLayers} cross-attn={string.Join(",", text.Config.CrossAttnLayers)}");
        Console.WriteLine($"Loading mmproj {Path.GetFileName(mmproj)} ...");
        using MllamaVisionEncoder vision = MllamaVisionEncoder.Load(mmproj);

        if (cuda is not null && Environment.GetEnvironmentVariable("HARTSY_VLM_NOPRELOAD") != "1")
            backend.PreloadWeights(text.Transformer.EnumerateWeights());

        int s = vision.ImageSize;
        int[] rgb = ParseColor(Environment.GetEnvironmentVariable("HARTSY_VLM_COLOR")) ?? [220, 40, 30];
        string pattern = Environment.GetEnvironmentVariable("HARTSY_VLM_PATTERN") ?? "circle";
        using Tensor px = SyntheticImage(s, rgb, pattern);
        Console.WriteLine($"  image = {pattern} rgb=({rgb[0]},{rgb[1]},{rgb[2]}), question: \"{question}\"");

        MllamaGenerator gen = new(text, vision, backend);
        System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
        string answer = gen.Generate(px, question, genTokens);
        sw.Stop();
        Console.WriteLine($"\n=== answer ({sw.Elapsed.TotalSeconds:F1}s) ===\n{answer.Trim()}");
        return 0;
    }

    /// <summary>A solid-background image with a centered filled shape of the given colour, normalized [1,3,S,S].</summary>
    private static unsafe Tensor SyntheticImage(int s, int[] rgb, string pattern)
    {
        Tensor t = new(new TensorShape(1, 3, s, s), DType.F32);
        float* p = (float*)t.DataPointer;
        int cx = s / 2, cy = s / 2, r = s / 3;
        for (int y = 0; y < s; y++)
            for (int x = 0; x < s; x++)
            {
                bool inside = pattern == "square"
                    ? Math.Abs(x - cx) < r && Math.Abs(y - cy) < r
                    : (x - cx) * (x - cx) + (y - cy) * (y - cy) < r * r;
                for (int c = 0; c < 3; c++)
                {
                    float v = (inside ? rgb[c] : 245) / 255f;      // shape colour on a near-white background
                    p[(long)c * s * s + (long)y * s + x] = (v - Mean[c]) / Std[c];
                }
            }
        return t;
    }

    private static int[]? ParseColor(string? str)
    {
        if (string.IsNullOrWhiteSpace(str)) return null;
        string[] parts = str.Split(',');
        if (parts.Length != 3) return null;
        int[] c = new int[3];
        for (int i = 0; i < 3; i++) if (!int.TryParse(parts[i].Trim(), out c[i])) return null;
        return c;
    }
}
