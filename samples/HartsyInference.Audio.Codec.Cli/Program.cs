using System.Diagnostics;
using HartsyInference.Audio.Cache;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Cuda;

namespace HartsyInference.Audio.Codec.Cli;

/// <summary>
/// Demo CLI for audio codecs: EnCodec, DAC, Mimi, SNAC, Vocos, WavTokenizer, BiCodec, XCodec, Oobleck.
/// Encodes/decodes audio and demonstrates neural codec properties.
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

        string mode = "encode";
        string codec = "encodec";
        string audioPath = "";
        int bitrate = 6;
        string backend = "cuda";

        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            if (a is "-m" or "--mode") { mode = args[++i].ToLowerInvariant(); }
            else if (a is "-c" or "--codec") { codec = args[++i].ToLowerInvariant(); }
            else if (a is "-b" or "--bitrate") { bitrate = int.Parse(args[++i]); }
            else if (a is "--backend") { backend = args[++i]; }
            else if (a.StartsWith('-')) { Console.Error.WriteLine($"unknown flag: {a}"); return 1; }
            else { audioPath = a; }
        }

        if (string.IsNullOrEmpty(audioPath))
        {
            Console.Error.WriteLine("missing AUDIO_PATH argument");
            PrintUsage();
            return 1;
        }

        if (!File.Exists(audioPath))
        {
            Console.Error.WriteLine($"Audio file not found: {audioPath}");
            return 1;
        }

        Console.Error.WriteLine($"mode:    {mode}");
        Console.Error.WriteLine($"codec:   {codec}");
        Console.Error.WriteLine($"audio:   {audioPath}");
        Console.Error.WriteLine($"bitrate: {bitrate} kbps (if applicable)");
        Console.Error.WriteLine($"backend: {backend}");
        Console.Error.WriteLine();

        using IBackend computeBackend = CreateBackend(backend);

        Console.Error.WriteLine("Audio Codec CLI for HartsyInference");
        Console.Error.WriteLine();
        Console.Error.WriteLine("Supported codecs:");
        Console.Error.WriteLine("  encodec        Meta EnCodec (24, 32 kHz)");
        Console.Error.WriteLine("  dac            DAC (16, 24 kHz, 8-16 kbps)");
        Console.Error.WriteLine("  mimi           Kyutai Mimi (16 kHz, 1-13.3 kbps)");
        Console.Error.WriteLine("  snac           Snake Audio Codec (16 kHz, hierarchical)");
        Console.Error.WriteLine("  vocos          Vocos (44.1 kHz, neural vocoder)");
        Console.Error.WriteLine("  wavtokenizer   Wav Tokenizer (16 kHz, semantic tokens)");
        Console.Error.WriteLine("  bicodec        BiCodec (16 kHz, bidirectional)");
        Console.Error.WriteLine("  xcodec         X Codec (24 kHz, compressible)");
        Console.Error.WriteLine("  oobleck        Oobleck (16-24 kHz, efficient)");
        Console.Error.WriteLine();
        Console.Error.WriteLine("Note: Full implementation coming soon!");
        Console.Error.WriteLine($"Mode {mode} on {Path.GetFileName(audioPath)} with {codec}");

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
        Console.WriteLine("HartsyInference Audio Codec CLI");
        Console.WriteLine();
        Console.WriteLine("Usage: HartsyInference.Audio.Codec.Cli -m <mode> [options] AUDIO_PATH");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  -m, --mode <mode>        Operation: encode, decode (default: encode)");
        Console.WriteLine("  -c, --codec <name>       Codec: encodec, dac, mimi, snac, vocos, wavtokenizer, etc.");
        Console.WriteLine("  -b, --bitrate <N>        Target bitrate in kbps (default: 6)");
        Console.WriteLine("  --backend <name>         Backend: cuda, cpu (default: cuda)");
        Console.WriteLine("  -h, --help               Show this help");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  HartsyInference.Audio.Codec.Cli -m encode -c encodec -b 6 audio.wav");
        Console.WriteLine("  HartsyInference.Audio.Codec.Cli -m decode -c dac compressed.codes");
        Console.WriteLine("  HartsyInference.Audio.Codec.Cli -m encode -c vocos -b 24 song.wav");
    }
}
