using SharpInference.Audio.Pipelines;
using SharpInference.Core.Tensors;

namespace SharpInference.Whisper.Cli;

/// <summary>F5-TTS smoke harness — loads the pipeline and runs a single forward pass on
/// tiny synthetic input to debug shape / weight-loading issues.</summary>
internal static class DebugF5
{
    public static async Task<int> RunAsync()
    {
        Console.Error.WriteLine("loading F5-TTS pipeline (~1.3 GB DiT + ~50 MB Vocos)...");
        using F5TtsPipeline pipe = await F5TtsPipeline.LoadAsync();
        using SharpInference.Cpu.CpuBackend backend = new();

        Console.Error.WriteLine("running smoke forward (T=8, text_len=5)...");
        try
        {
            Tensor v = pipe.SmokeForward(backend, t: 8, textLen: 5);
            Console.Error.WriteLine($"OK — output shape: [{string.Join(",", Enumerable.Range(0, v.Shape.Rank).Select(i => v.Shape[i].ToString()))}]");
            unsafe
            {
                float* p = (float*)v.DataPointer;
                Console.Error.WriteLine($"first 5: {p[0]:F4}, {p[1]:F4}, {p[2]:F4}, {p[3]:F4}, {p[4]:F4}");
                double sumAbs = 0;
                for (long i = 0; i < v.ElementCount; i++) sumAbs += Math.Abs(p[i]);
                Console.Error.WriteLine($"mean abs: {sumAbs / v.ElementCount:E3}");
            }
            v.Dispose();
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"FAILED: {ex.GetType().Name}: {ex.Message}");
            Console.Error.WriteLine(ex.StackTrace);
            return 1;
        }
    }
}
