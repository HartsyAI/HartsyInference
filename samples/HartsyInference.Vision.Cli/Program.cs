using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Cuda;

namespace HartsyInference.Vision.Cli;

/// <summary>
/// Demo CLI for vision tasks: CLIP embeddings, YOLO detection, SAM segmentation, face detection.
/// Accepts images and outputs embeddings/detections as JSON or annotated images.
/// </summary>
internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help")
        {
            PrintUsage();
            return args.Length == 0 ? 1 : 0;
        }

        string mode = "embed";
        string model = "clip-vit-l-14";
        string imagePath = "";
        float confidence = 0.5f;
        string backend = "cuda";

        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            if (a is "-m" or "--mode") { mode = args[++i].ToLowerInvariant(); }
            else if (a is "--model") { model = args[++i].ToLowerInvariant(); }
            else if (a is "-c" or "--confidence") { confidence = float.Parse(args[++i]); }
            else if (a is "-b" or "--backend") { backend = args[++i]; }
            else if (a.StartsWith('-')) { Console.Error.WriteLine($"unknown flag: {a}"); return 1; }
            else { imagePath = a; }
        }

        if (string.IsNullOrEmpty(imagePath))
        {
            Console.Error.WriteLine("missing IMAGE_PATH argument");
            PrintUsage();
            return 1;
        }

        if (!File.Exists(imagePath))
        {
            Console.Error.WriteLine($"Image file not found: {imagePath}");
            return 1;
        }

        Console.Error.WriteLine($"mode:       {mode}");
        Console.Error.WriteLine($"model:      {model}");
        Console.Error.WriteLine($"image:      {imagePath}");
        Console.Error.WriteLine($"confidence: {confidence}");
        Console.Error.WriteLine($"backend:    {backend}");
        Console.Error.WriteLine();

        using IBackend computeBackend = CreateBackend(backend);

        Console.Error.WriteLine("Vision CLI for HartsyInference");
        Console.Error.WriteLine();
        Console.Error.WriteLine("Modes:");
        Console.Error.WriteLine("  embed    Image embeddings (CLIP, SigLIP)");
        Console.Error.WriteLine("  detect   Object detection (YOLO8, YOLO11)");
        Console.Error.WriteLine("  segment  Instance segmentation (SAM, SAM2)");
        Console.Error.WriteLine("  face     Face detection + landmarks (RetinaFace)");
        Console.Error.WriteLine();
        Console.Error.WriteLine("Note: Full implementation coming soon!");
        Console.Error.WriteLine($"Mode {mode} on {Path.GetFileName(imagePath)} with {model}");

        return 0;
    }

    private static IBackend CreateBackend(string backendName) => backendName switch
    {
        "cuda" => new CudaBackend(deviceOrdinal: 0, ptxDir: Path.Combine(AppContext.BaseDirectory, "Ptx")),
        "cpu" => new CpuBackend(),
        _ => throw new ArgumentException($"Unknown backend '{backendName}'. Valid: cpu, cuda."),
    };

    private static void PrintUsage()
    {
        Console.WriteLine("HartsyInference Vision CLI");
        Console.WriteLine();
        Console.WriteLine("Usage: HartsyInference.Vision.Cli -m <mode> [options] IMAGE_PATH");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  -m, --mode <mode>        Task mode: embed, detect, segment, face (default: embed)");
        Console.WriteLine("  --model <model>          Model name (varies by mode)");
        Console.WriteLine("  -c, --confidence <N>     Confidence threshold (default: 0.5)");
        Console.WriteLine("  -b, --backend <name>     Backend: cuda, cpu (default: cuda)");
        Console.WriteLine("  -h, --help               Show this help");
        Console.WriteLine();
        Console.WriteLine("Embedding models (--mode embed):");
        Console.WriteLine("  clip-vit-l-14   CLIP ViT-L/14");
        Console.WriteLine("  clip-vit-h-14   CLIP ViT-H/14");
        Console.WriteLine("  siglip-so       SigLIP SO");
        Console.WriteLine("  dinov2-large    DINOv2 Large");
        Console.WriteLine();
        Console.WriteLine("Detection models (--mode detect):");
        Console.WriteLine("  yolo8-n         YOLO8 Nano");
        Console.WriteLine("  yolo8-m         YOLO8 Medium");
        Console.WriteLine("  yolo11-xl       YOLO11 Extra Large");
        Console.WriteLine();
        Console.WriteLine("Segmentation models (--mode segment):");
        Console.WriteLine("  sam-vit-h       SAM ViT-H");
        Console.WriteLine("  sam2            SAM 2");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  HartsyInference.Vision.Cli -m embed image.jpg");
        Console.WriteLine("  HartsyInference.Vision.Cli -m detect --model yolo11-xl photo.png -c 0.6");
        Console.WriteLine("  HartsyInference.Vision.Cli -m face portrait.jpg");
    }
}
