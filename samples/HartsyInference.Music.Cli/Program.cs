using System.Diagnostics;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Cuda;

namespace HartsyInference.Music.Cli;

/// <summary>
/// Demo CLI for music generation: ACE-Step, MusicGen, YuE.
/// Generates audio from text prompts and saves as WAV.
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

        string model = "ace-step";
        string prompt = "upbeat electronic dance music";
        int durationSeconds = 30;
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

        Console.Error.WriteLine("Music generation CLI for HartsyInference");
        Console.Error.WriteLine();
        Console.Error.WriteLine("Supported models:");
        Console.Error.WriteLine("  ace-step    ACE-Step (flow-matching, high quality, ~30s)");
        Console.Error.WriteLine("  musicgen    MusicGen (transformer, multi-style, ~30s)");
        Console.Error.WriteLine("  yue         YuE (dual-stage Llama, coherent, experimental)");
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
        Console.WriteLine("HartsyInference Music Generation CLI");
        Console.WriteLine();
        Console.WriteLine("Usage: HartsyInference.Music.Cli [options]");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  -m, --model <model>      Model: ace-step, musicgen, yue (default: ace-step)");
        Console.WriteLine("  -p, --prompt \"...\"       Music description (default: 'upbeat electronic')");
        Console.WriteLine("  -d, --duration <N>       Duration in seconds (default: 30)");
        Console.WriteLine("  -s, --seed <N>           RNG seed (default: 42)");
        Console.WriteLine("  -b, --backend <name>     Backend: cuda, cpu (default: cuda)");
        Console.WriteLine("  -h, --help               Show this help");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  HartsyInference.Music.Cli -m ace-step -p \"calm ambient music\"");
        Console.WriteLine("  HartsyInference.Music.Cli -m musicgen -d 60 -p \"jazz piano\"");
        Console.WriteLine("  HartsyInference.Music.Cli -m yue -p \"lo-fi hip-hop beats\" -b cpu");
    }
}
