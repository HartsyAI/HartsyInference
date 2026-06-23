using System.Diagnostics;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Cuda;

namespace HartsyInference.Speech.Cli;

/// <summary>
/// Demo CLI for text-to-speech: Kokoro, F5-TTS, VibeVoice, CosyVoice, Bark.
/// Converts text to speech and saves as WAV.
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

        string model = "kokoro";
        string text = "Hello, this is a text-to-speech demonstration.";
        string voice = "default";
        float speed = 1.0f;
        string backend = "cuda";

        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            if (a is "-m" or "--model") { model = args[++i]; }
            else if (a is "-t" or "--text") { text = args[++i]; }
            else if (a is "-v" or "--voice") { voice = args[++i]; }
            else if (a is "--speed") { speed = float.Parse(args[++i]); }
            else if (a is "-b" or "--backend") { backend = args[++i]; }
            else if (a.StartsWith('-')) { Console.Error.WriteLine($"unknown flag: {a}"); return 1; }
        }

        Console.Error.WriteLine($"model:   {model}");
        Console.Error.WriteLine($"text:    \"{text}\"");
        Console.Error.WriteLine($"voice:   {voice}");
        Console.Error.WriteLine($"speed:   {speed}x");
        Console.Error.WriteLine($"backend: {backend}");
        Console.Error.WriteLine();

        using IBackend computeBackend = CreateBackend(backend);

        Console.Error.WriteLine("Text-to-Speech CLI for HartsyInference");
        Console.Error.WriteLine();
        Console.Error.WriteLine("Supported models:");
        Console.Error.WriteLine("  kokoro      Kokoro (fast, natural, 24 kHz)");
        Console.Error.WriteLine("  f5-tts      F5-TTS (voice cloning, diffusion)");
        Console.Error.WriteLine("  vibevoice   VibeVoice (diffusion-based, expressive)");
        Console.Error.WriteLine("  cosyvoice   CosyVoice (Qwen-based LM, natural)");
        Console.Error.WriteLine("  bark        Bark (text + style tokens, unique)");
        Console.Error.WriteLine();
        Console.Error.WriteLine("Note: Full implementation coming soon!");
        Console.Error.WriteLine($"Model {model} with text '{text}' at {speed}x speed");

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
        Console.WriteLine("HartsyInference Text-to-Speech CLI");
        Console.WriteLine();
        Console.WriteLine("Usage: HartsyInference.Speech.Cli [options]");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  -m, --model <model>      Model: kokoro, f5-tts, vibevoice, cosyvoice, bark (default: kokoro)");
        Console.WriteLine("  -t, --text \"...\"         Text to synthesize");
        Console.WriteLine("  -v, --voice <name>       Voice/speaker ID (model-dependent)");
        Console.WriteLine("  --speed <N>              Playback speed multiplier (default: 1.0)");
        Console.WriteLine("  -b, --backend <name>     Backend: cuda, cpu (default: cuda)");
        Console.WriteLine("  -h, --help               Show this help");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  HartsyInference.Speech.Cli -m kokoro -t \"Hello world\"");
        Console.WriteLine("  HartsyInference.Speech.Cli -m f5-tts -t \"This is voice cloning\" -v reference.wav");
        Console.WriteLine("  HartsyInference.Speech.Cli -m bark -t \"This is BARK!\" --speed 1.5");
    }
}
