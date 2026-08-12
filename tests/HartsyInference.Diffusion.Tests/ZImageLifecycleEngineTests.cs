using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Persistent-process regression for Z-Image's Qwen → DiT → VAE phase ownership. Opt-in because it
/// loads roughly 15 GB of real checkpoint data and exercises successful, cancelled, prompt-miss, and prompt-hit
/// generations on one engine.</summary>
[Collection("CudaSerial")]
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class ZImageLifecycleEngineTests
{
    private readonly ITestOutputHelper _output;

    public ZImageLifecycleEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task RepeatedGenerations_InOneEngine_DifferentPrompt_NoVramAccumulation()
    {
        if (Environment.GetEnvironmentVariable("HARTSY_RUN_ZIMAGE_LIFECYCLE_TEST") != "1")
        {
            _output.WriteLine("SKIPPED: set HARTSY_RUN_ZIMAGE_LIFECYCLE_TEST=1 for the real-weight lifecycle gate.");
            return;
        }
        if (Environment.GetEnvironmentVariable("HARTSY_KEEP_MODELS") != "0")
            throw new InvalidOperationException("This regression must run in a fresh process with HARTSY_KEEP_MODELS=0.");
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        string checkpoint = TestPaths.ZImage.Turbo;
        string qwen = TestPaths.TextEncoders.Qwen3_4B;
        string vae = Path.Combine(TestPaths.ModelsDir, "VAE", "Flux", "ae.safetensors");
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint, qwen, vae))
            return;

        ModelSpec spec = ModelResolver.Resolve("zimage", checkpoint, Modality.Image);
        ImageRequest firstRequest = Request(
            "A red fox wearing a cobalt blue scarf, sitting in fresh snow beneath a pine tree at golden hour",
            seed: 424242);
        ImageRequest secondRequest = Request(
            "A green ceramic teapot beside three yellow lemons on a walnut table, soft window light, product photograph",
            seed: 424243);

        using InferenceEngine engine = new("cuda", 0);
        _output.WriteLine("[1/4] first prompt — constructs and drives Qwen, DiT, and VAE");
        ImageResult first = await engine.Images.GenerateAsync(spec, firstRequest);
        AssertCoherent(first, "first");
        long freeAfterFirst = StableFreeBytes(engine);
        _output.WriteLine($"free after first generation: {freeAfterFirst / (1024.0 * 1024.0 * 1024.0):F2} GiB");

        _output.WriteLine("[2/4] cancel at first denoise progress callback — rollback must leave the engine reusable");
        using (CancellationTokenSource cancel = new())
        {
            ImageRequest cancelledRequest = Request(
                "A copper weather vane above a red barn during a summer thunderstorm",
                seed: 424244);
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => engine.Images.GenerateAsync(
                spec,
                cancelledRequest,
                new CancelOnFirstStep(cancel),
                cancel.Token));
        }
        long freeAfterCancellation = StableFreeBytes(engine);
        _output.WriteLine($"free after cancelled generation: {freeAfterCancellation / (1024.0 * 1024.0 * 1024.0):F2} GiB");
        long cancellationRetention = freeAfterFirst - freeAfterCancellation;
        Assert.True(cancellationRetention < 512L * 1024 * 1024,
            $"Z-Image retained {cancellationRetention / (1024.0 * 1024.0):F0} MiB after cancellation.");

        _output.WriteLine("[3/4] different prompt — must reload Qwen and complete after cancellation in the same process");
        ImageResult second = await engine.Images.GenerateAsync(spec, secondRequest);
        AssertCoherent(second, "second");
        long freeAfterSecond = StableFreeBytes(engine);
        _output.WriteLine($"free after second generation: {freeAfterSecond / (1024.0 * 1024.0 * 1024.0):F2} GiB");

        long drift = freeAfterFirst - freeAfterSecond;
        _output.WriteLine($"free-memory drift: {drift / (1024.0 * 1024.0):F0} MiB");
        Assert.True(Math.Abs(drift) < 512L * 1024 * 1024,
            $"Z-Image retained {drift / (1024.0 * 1024.0):F0} MiB across the second generation.");

        _output.WriteLine("[4/4] same prompt — must hit the prompt cache while KEEP_MODELS=0 still evicts post-denoise state");
        ImageResult third = await engine.Images.GenerateAsync(spec, secondRequest);
        AssertCoherent(third, "third");
        Assert.Equal(second.Rgb, third.Rgb);
        long freeAfterThird = StableFreeBytes(engine);
        _output.WriteLine($"free after third generation: {freeAfterThird / (1024.0 * 1024.0 * 1024.0):F2} GiB");
        long cacheHitDrift = freeAfterSecond - freeAfterThird;
        Assert.True(Math.Abs(cacheHitDrift) < 512L * 1024 * 1024,
            $"Z-Image retained {cacheHitDrift / (1024.0 * 1024.0):F0} MiB across a prompt-cache-hit generation.");
    }

    private static ImageRequest Request(string prompt, int seed) => new()
    {
        Prompt = prompt,
        Width = 512,
        Height = 512,
        Steps = 2,
        CfgScale = 1.0f,
        Seed = seed,
    };

    private static long StableFreeBytes(InferenceEngine engine)
    {
        engine.ComputeBackend.Sync();
        engine.ComputeBackend.TrimMemoryPool();
        return engine.ComputeBackend.FreeMemoryBytes();
    }

    private static void AssertCoherent(ImageResult result, string label)
    {
        Assert.Equal(512, result.Width);
        Assert.Equal(512, result.Height);
        Assert.Equal(512 * 512 * 3, result.Rgb.Length);
        int nonZero = result.Rgb.Count(value => value != 0);
        int nonWhite = result.Rgb.Count(value => value != 255);
        Assert.True(nonZero > result.Rgb.Length / 10, $"{label} image is effectively all black.");
        Assert.True(nonWhite > result.Rgb.Length / 10, $"{label} image is effectively all white.");
    }

    private sealed class CancelOnFirstStep(CancellationTokenSource cancel) : IProgress<StepPreview>
    {
        private int _reports;

        public void Report(StepPreview value)
        {
            if (Interlocked.Increment(ref _reports) != 1)
                return;
            cancel.Cancel();
            cancel.Token.ThrowIfCancellationRequested();
        }
    }
}
