using System.Diagnostics;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;

// dispatch helpers below


namespace HartsyInference.Whisper.Cli;

/// <summary>Demo CLI: download a Whisper checkpoint from HuggingFace, run it on a
/// WAV file, print the transcription. Mostly exists as living documentation —
/// production server / batch usage should call <see cref="WhisperPipeline"/> directly.</summary>
internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help")
        {
            PrintUsage();
            return args.Length == 0 ? 1 : 0;
        }

        string? wavPath = null;
        string model = "openai/whisper-tiny";
        string language = "en";
        bool translate = false;
        bool timestamps = false;

        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            if (a is "-m" or "--model") { model = args[++i]; }
            else if (a is "-l" or "--language") { language = args[++i]; }
            else if (a is "-t" or "--translate") { translate = true; }
            else if (a is "--timestamps") { timestamps = true; }
            else if (a.StartsWith('-')) { Console.Error.WriteLine($"unknown flag: {a}"); return 1; }
            else { wavPath = a; }
        }

        // Dispatch to the Moonshine pipeline if the user picked a Moonshine repo.
        if (model.StartsWith("UsefulSensors/", StringComparison.OrdinalIgnoreCase))
            return await RunMoonshineAsync(wavPath, model);




        if (wavPath is null)
        {
            Console.Error.WriteLine("missing WAV file argument");
            PrintUsage();
            return 1;
        }
        if (!File.Exists(wavPath))
        {
            Console.Error.WriteLine($"WAV file not found: {wavPath}");
            return 1;
        }

        Console.Error.WriteLine($"model:    {model}");
        Console.Error.WriteLine($"cache:    {AudioModelCache.CacheRoot}");
        Console.Error.WriteLine($"language: {language}");
        Console.Error.WriteLine($"task:     {(translate ? "translate" : "transcribe")}");
        Console.Error.WriteLine($"wav:      {wavPath}");
        Console.Error.WriteLine();

        Stopwatch sw = Stopwatch.StartNew();
        Console.Error.WriteLine("loading pipeline...");
        using WhisperPipeline pipeline = await WhisperPipeline.LoadAsync(model);
        Console.Error.WriteLine($"  loaded in {sw.Elapsed.TotalSeconds:F1}s");

        // We default to the CPU backend; CUDA users can flip to CudaBackend in a
        // real application. The pipeline doesn't care which backend is passed.
        using IBackend backend = new HartsyInference.Cpu.CpuBackend();

        WhisperOptions options = new()
        {
            Language = language,
            Translate = translate,
            WithTimestamps = timestamps,
        };

        sw.Restart();
        Console.Error.WriteLine("transcribing...");
        string text = pipeline.TranscribeWav(backend, wavPath, options);
        Console.Error.WriteLine($"  decoded in {sw.Elapsed.TotalSeconds:F1}s");
        Console.Error.WriteLine();

        Console.WriteLine(text);
        return 0;
    }

    private static async Task<int> RunMoonshineAsync(string? wavPath, string model)
    {
        if (wavPath is null) { Console.Error.WriteLine("missing WAV file argument"); return 1; }
        if (!File.Exists(wavPath)) { Console.Error.WriteLine($"WAV file not found: {wavPath}"); return 1; }

        Console.Error.WriteLine($"model:    {model}");
        Console.Error.WriteLine($"cache:    {AudioModelCache.CacheRoot}");
        Console.Error.WriteLine($"wav:      {wavPath}");
        Console.Error.WriteLine();

        Stopwatch sw = Stopwatch.StartNew();
        Console.Error.WriteLine("loading Moonshine pipeline...");
        using MoonshinePipeline pipeline = await MoonshinePipeline.LoadAsync(model);
        Console.Error.WriteLine($"  loaded in {sw.Elapsed.TotalSeconds:F1}s");

        using IBackend backend = new HartsyInference.Cpu.CpuBackend();

        sw.Restart();
        Console.Error.WriteLine("transcribing...");
        string text = pipeline.TranscribeWav(backend, wavPath);
        Console.Error.WriteLine($"  decoded in {sw.Elapsed.TotalSeconds:F1}s");
        Console.Error.WriteLine();

        Console.WriteLine(text);
        return 0;
    }

    private static void PrintUsage()
    {
        Console.Error.WriteLine("""
            Usage: hartsyinference-whisper [options] <wav-file>

            Options:
              -m, --model <repo>      HuggingFace repo id  (default: openai/whisper-tiny)
                                      One of: openai/whisper-{tiny,base,small,medium,large-v2,large-v3,large-v3-turbo}
                                              distil-whisper/distil-{small.en,medium.en,large-v2,large-v3}
              -l, --language <code>   Two-letter language code (default: en)
              -t, --translate         Translate to English instead of transcribing in source language
                  --timestamps        Emit timestamp tokens in the output
              -h, --help              Show this message

            Models are downloaded to ~/.cache/hartsyinference/models/ on first use.
            Audio is auto-resampled to 16 kHz and downmixed to mono.

            Example:
              hartsyinference-whisper sample.wav
              hartsyinference-whisper -m openai/whisper-large-v3-turbo -l ja --translate clip.wav
            """);
    }
}
