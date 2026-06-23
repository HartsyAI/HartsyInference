using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Cuda;

namespace HartsyInference.Video.Cli;

/// <summary>
/// Demo CLI for video generation: LTX-Video, Wan 2.2, Lance, Kandinsky Video.
/// Generates video from text prompts and saves as MP4.
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

        string model = "ltx-video";
        string prompt = "a cat walking across a sunny garden";
        int durationSeconds = 8;
        int seed = 42;
        string backend = "cuda";

        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            if (a is "-m" or "--model") { model = args[++i]; }
            else if (a is "-p" or "--prompt") { prompt = args[++i]; }
            else if (a is "-d" or "--duration") { durationSeconds = int.Parse(args[++i]); }
            else if (a is "-s" or "--seed") { seed = int.Parse(args[++i]); }
            else if (a is "-b" or "--backend") { backend = args[++i]; }
            else if (a.StartsWith('-')) { Console.Error.WriteLine($"unknown flag: {a}"); return 1; }
        }

        Console.Error.WriteLine($"model:    {model}");
        Console.Error.WriteLine($"prompt:   \"{prompt}\"");
        Console.Error.WriteLine($"duration: {durationSeconds}s");
        Console.Error.WriteLine($"seed:     {seed}");
        Console.Error.WriteLine($"backend:  {backend}");
        Console.Error.WriteLine();

        using IBackend computeBackend = CreateBackend(backend);

        Console.Error.WriteLine("Video Generation CLI for HartsyInference");
        Console.Error.WriteLine();
        Console.Error.WriteLine("Supported models:");
        Console.Error.WriteLine("  ltx-video   LTX-Video (768p, up to 32 frames)");
        Console.Error.WriteLine("  wan         Wan 2.2 (video diffusion, 480p-1080p)");
        Console.Error.WriteLine("  lance       Lance (3B active, 480p, 4-121 frames)");
        Console.Error.WriteLine("  kandinsky   Kandinsky Video (1024p, video generation)");
        Console.Error.WriteLine();
        Console.Error.WriteLine("Note: Full implementation coming soon!");
        Console.Error.WriteLine($"Model {model} with prompt '{prompt}' at {durationSeconds}s");

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
        Console.WriteLine("HartsyInference Video Generation CLI");
        Console.WriteLine();
        Console.WriteLine("Usage: HartsyInference.Video.Cli [options]");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  -m, --model <model>      Model: ltx-video, wan, lance, kandinsky (default: ltx-video)");
        Console.WriteLine("  -p, --prompt \"...\"       Video description");
        Console.WriteLine("  -d, --duration <N>       Duration in seconds (default: 8)");
        Console.WriteLine("  -s, --seed <N>           RNG seed (default: 42)");
        Console.WriteLine("  -b, --backend <name>     Backend: cuda, cpu (default: cuda)");
        Console.WriteLine("  -h, --help               Show this help");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  HartsyInference.Video.Cli -m ltx-video -p \"dog running in park\" -d 8");
        Console.WriteLine("  HartsyInference.Video.Cli -m wan -p \"ocean waves crashing\" -d 16");
        Console.WriteLine("  HartsyInference.Video.Cli -m lance -p \"flying through mountains\" -d 30");
    }
}
