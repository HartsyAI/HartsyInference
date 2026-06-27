using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Multimodal;

namespace HartsyInference.TextGen.Cli;

/// <summary>Vision-language smoke harness for Gemma-3 VLMs. Feeds a synthetic solid-colour image (so the answer
/// is checkable without an image decoder) and a question through the full pipeline:
/// SigLIP vision tower → Gemma-3 projector → splice into the text decoder → greedy decode.</summary>
public static class VlmRunner
{
    public static unsafe int Run(string backendName, int genTokens, string question)
    {
        string? textPath = Environment.GetEnvironmentVariable("HARTSY_MODEL_DIR");
        string? mmprojPath = Environment.GetEnvironmentVariable("HARTSY_MMPROJ");
        if (textPath is null || !File.Exists(textPath)) { Console.Error.WriteLine("Set HARTSY_MODEL_DIR to the Gemma-3 text .gguf."); return 1; }
        if (mmprojPath is null || !File.Exists(mmprojPath)) { Console.Error.WriteLine("Set HARTSY_MMPROJ to the mmproj-*.gguf."); return 1; }

        using IBackend backend = backendName == "cpu"
            ? new CpuBackend()
            : new CudaBackend(deviceOrdinal: 0, ptxDir: Path.Combine(AppContext.BaseDirectory, "Ptx"));
        CudaBackend? cuda = backend as CudaBackend;
        bool lowVram = Environment.GetEnvironmentVariable("HARTSY_LOWVRAM") == "1";

        Console.WriteLine($"=== VLM (Gemma-3) — backend={backendName} ===");
        Console.WriteLine($"Loading text {Path.GetFileName(textPath)} ...");
        using GgufLanguageModel text = GgufLanguageModel.Load(textPath, lowVram, dequantizeToF32: cuda is null);
        Console.WriteLine($"  text arch={text.Architecture} hidden={text.Config.HiddenSize} layers={text.Config.NumLayers}");
        Console.WriteLine($"Loading mmproj {Path.GetFileName(mmprojPath)} ...");
        // Qwen2.5-VL uses its own windowed 2D-RoPE ViT; everything else (Gemma-3/SmolVLM/LLaVA) is SigLIP/CLIP.
        bool isQwen = IsQwen25Vl(mmprojPath);
        using IVlmImageEncoder vision = isQwen ? Qwen25VlEncoder.Load(mmprojPath) : SiglipVlmEncoder.Load(mmprojPath);
        Console.WriteLine($"  vision family={vision.Family} image={vision.ImageSize} tokens/img={vision.TokensPerImage} text_dim={(vision is SiglipVlmEncoder s ? s.ProjectionDim : ((Qwen25VlEncoder)vision).ProjectionDim)}");

        // Encode-only dump/debug doesn't run the decoder, so skip the (large) text-weight GPU preload.
        // HARTSY_VLM_NOPRELOAD also skips it (slower per-op uploads, but fits when the GPU is shared with another app).
        if (cuda is not null && Environment.GetEnvironmentVariable("HARTSY_VLM_ENCODE_ONLY") != "1"
            && Environment.GetEnvironmentVariable("HARTSY_VLM_NOPRELOAD") != "1")
        {
            backend.PreloadWeights(text.Transformer.EnumerateWeights());
        }

        // Synthetic structured test image → normalized pixel values [1, 3, S, S]. Solid colours are a poor VLM
        // test (spatially uniform → out-of-distribution for SigLIP), so default to a recognizable shape.
        // HARTSY_VLM_PNG: decode a real PNG → bilinear-resize + normalize via VlmImagePreprocessor (production path).
        string? pngPath = Environment.GetEnvironmentVariable("HARTSY_VLM_PNG");
        int imgSize = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_VLM_IMGSIZE"), out int ov) ? ov : vision.ImageSize;
        if (pngPath is not null)
        {
            (byte[] rgb, int w, int h) = HartsyInference.Vision.Codec.PngDecoder.DecodeFromFile(pngPath);
            using Tensor pix = VlmImagePreprocessor.Preprocess(rgb, w, h, imgSize, vision.ImageMean, vision.ImageStd);
            Console.WriteLine($"  image = PNG {Path.GetFileName(pngPath)} ({w}x{h} → {imgSize}), question: \"{question}\"");
            MultimodalGenerator g2 = new(text, vision, backend);
            Stopwatch s2 = Stopwatch.StartNew();
            string a2 = g2.Generate(pix, question, genTokens);
            backend.Sync(); s2.Stop();
            Console.WriteLine($"\n=== answer ({s2.Elapsed.TotalSeconds:F1}s) ===\n{a2}");
            return 0;
        }

        // HARTSY_VLM_IMAGE_FILE: load a raw [3, S, S] F32 (already-normalized) image, e.g. the reference's ref_image.f32,
        // so C# and the torch reference use a byte-identical input. Otherwise build a synthetic normalized pattern.
        string? imgFile = Environment.GetEnvironmentVariable("HARTSY_VLM_IMAGE_FILE");
        int S = imgFile is not null ? (int)Math.Round(Math.Sqrt(new FileInfo(imgFile).Length / 4.0 / 3.0)) : imgSize;
        using Tensor pixels = new(new TensorShape(1, 3, S, S), DType.F32);
        float* px = (float*)pixels.DataPointer;
        if (imgFile is not null)
        {
            byte[] raw = File.ReadAllBytes(imgFile);
            fixed (byte* rp = raw) Buffer.MemoryCopy(rp, px, raw.Length, raw.Length);
            Console.WriteLine($"  image = file {Path.GetFileName(imgFile)} ({S}x{S}), question: \"{question}\"");
        }
        else
        {
            string pattern = Environment.GetEnvironmentVariable("HARTSY_VLM_PATTERN") ?? "circle";
            int[] rgb = ParseColor(Environment.GetEnvironmentVariable("HARTSY_VLM_COLOR")) ?? [220, 30, 30];
            byte[,,] img = BuildImage(pattern, S, rgb);   // [3, S, S] in 0..255
            for (int c = 0; c < 3; c++)
            {
                float* chan = px + (long)c * S * S;
                for (int y = 0; y < S; y++)
                    for (int x = 0; x < S; x++)
                        chan[y * S + x] = (img[c, y, x] / 255f - vision.ImageMean[c]) / vision.ImageStd[c];
            }
            Console.WriteLine($"  image = pattern '{pattern}', question: \"{question}\"");
        }

        MultimodalGenerator gen = new(text, vision, backend);
        Stopwatch sw = Stopwatch.StartNew();
        string answer = gen.Generate(pixels, question, genTokens);
        backend.Sync();
        sw.Stop();
        Console.WriteLine($"\n=== answer ({sw.Elapsed.TotalSeconds:F1}s) ===\n{answer}");
        return 0;
    }

    /// <summary>Builds a simple structured RGB test image [3, S, S] (0..255) with clear, describable content.</summary>
    private static byte[,,] BuildImage(string pattern, int S, int[] fg)
    {
        byte[,,] img = new byte[3, S, S];
        void Set(int y, int x, int r, int g, int b) { img[0, y, x] = (byte)r; img[1, y, x] = (byte)g; img[2, y, x] = (byte)b; }
        switch (pattern)
        {
            case "halves":   // top sky-blue, bottom grass-green (a simple landscape)
                for (int y = 0; y < S; y++) for (int x = 0; x < S; x++)
                    if (y < S / 2) Set(y, x, 110, 170, 230); else Set(y, x, 60, 160, 60);
                break;
            case "checker":
                for (int y = 0; y < S; y++) for (int x = 0; x < S; x++)
                { bool on = ((x / (S / 8)) + (y / (S / 8))) % 2 == 0; int v = on ? 245 : 15; Set(y, x, v, v, v); }
                break;
            case "square":   // filled coloured square centred on white
                for (int y = 0; y < S; y++) for (int x = 0; x < S; x++)
                { bool ins = x > S / 4 && x < 3 * S / 4 && y > S / 4 && y < 3 * S / 4; if (ins) Set(y, x, fg[0], fg[1], fg[2]); else Set(y, x, 255, 255, 255); }
                break;
            default:          // "circle": filled coloured disk centred on white background
                int cx = S / 2, cy = S / 2, rad = S / 3;
                for (int y = 0; y < S; y++) for (int x = 0; x < S; x++)
                { bool ins = (x - cx) * (x - cx) + (y - cy) * (y - cy) <= rad * rad; if (ins) Set(y, x, fg[0], fg[1], fg[2]); else Set(y, x, 255, 255, 255); }
                break;
        }
        return img;
    }

    /// <summary>True if the mmproj is a Qwen2.5-VL merger (filename heuristic for the harness; production detects via
    /// the <c>clip.projector_type=qwen2.5vl_merger</c> metadata).</summary>
    private static bool IsQwen25Vl(string mmprojPath)
        => Path.GetFileName(mmprojPath).Replace(".", "").Replace("-", "").Contains("qwen25vl", StringComparison.OrdinalIgnoreCase)
        || Path.GetFileName(mmprojPath).Contains("Qwen2.5-VL", StringComparison.OrdinalIgnoreCase);

    private static int[]? ParseColor(string? s)
    {
        if (string.IsNullOrWhiteSpace(s)) return null;
        string[] parts = s.Split(',');
        if (parts.Length != 3) return null;
        int[] c = new int[3];
        for (int i = 0; i < 3; i++) if (!int.TryParse(parts[i].Trim(), out c[i])) return null;
        return c;
    }
}
