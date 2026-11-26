using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Device reads of the video and avatar denoise loops on CUDA. The counter is compared across a three-step run and a six-step run, after one
/// unmeasured warm-up run, so one-time costs do not land in only one of the two measurements. Each guided step is budgeted at four reads: the
/// forwards (one read each in steady state, two for video and three for the avatar's three passes), plus the CFG-zero sums for video, plus one
/// read of slack. The fourth read is not yet attributed (see the PR thread), so it is slack, not a known cost: if a later change removes it, the
/// budget should tighten to three. Returns without asserting when CUDA is not available, and that case is reported as a skip in the output.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class LongCatDeviceResidencyGpuTests
{
    private const int Frames = 3, Height = 4, Width = 4, TextLength = 6;
    private const int BudgetPerStep = 4;
    private static readonly int Hw = Height * Width;

    [Fact]
    public void AVideoRunReadsWithinItsForwardsAndSumsBudget()
    {
        if (!CudaContext.IsAvailable())
        {
            Console.WriteLine("SKIPPED: CUDA unavailable; the device-read budget was not measured.");
            return;
        }
        LongCatConfig config = LongCatTestData.Tiny;
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(config, 97);
        try
        {
            using CudaBackend cuda = new(0, PtxDirectory());
            using LongCatTransformer transformer = new(config);
            transformer.LoadWeights(weights);
            _ = CountVideoReads(cuda, transformer, config, steps: 1);
            long shortRun = CountVideoReads(cuda, transformer, config, steps: 3);
            long longRun = CountVideoReads(cuda, transformer, config, steps: 6);

            Assert.True(longRun - shortRun <= 3 * BudgetPerStep, $"{longRun - shortRun} reads over three extra steps");
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }

    [Fact]
    public void AnAvatarCachedRunReadsWithinItsForwardsBudget()
    {
        if (!CudaContext.IsAvailable())
        {
            Console.WriteLine("SKIPPED: CUDA unavailable; the device-read budget was not measured.");
            return;
        }
        LongCatConfig config = LongCatTestData.TinyAvatar;
        Dictionary<string, Tensor> weights = LongCatTestData.AvatarWeights(config, 101);
        try
        {
            using CudaBackend cuda = new(0, PtxDirectory());
            using LongCatTransformer transformer = new(config);
            transformer.LoadWeights(weights);
            _ = CountAvatarReads(cuda, transformer, config, steps: 1);
            long shortRun = CountAvatarReads(cuda, transformer, config, steps: 3);
            long longRun = CountAvatarReads(cuda, transformer, config, steps: 6);

            Assert.True(longRun - shortRun <= 3 * BudgetPerStep, $"{longRun - shortRun} reads over three extra steps");
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }

    private static long CountVideoReads(CudaBackend cuda, LongCatTransformer transformer, LongCatConfig config, int steps)
    {
        Random rng = new(23);
        LongCatPrompt positive = new(LongCatTestData.RandomValues(rng, TextLength * config.CaptionChannels), TextLength, TextLength);
        LongCatPrompt negative = new(LongCatTestData.RandomValues(rng, TextLength * config.CaptionChannels), TextLength, TextLength);
        LongCatRun run = new(config.InChannels, Frames, Height, Width, LongCatTestData.RandomValues(rng, config.InChannels * Frames * Hw), positive,
            negative, steps, Distill: false, Shift: 3f, Guidance: 4f);
        cuda.ResetD2hSyncCount();
        new LongCatVideoPipeline(cuda, transformer).Denoise(run);
        return cuda.GetD2hSyncCount();
    }

    private static long CountAvatarReads(CudaBackend cuda, LongCatTransformer transformer, LongCatConfig config, int steps)
    {
        Random rng = new(29);
        int cond = 2, ctx = config.AudioContextTokens, width = config.AudioOutputDim;
        LongCatPrompt positive = new(LongCatTestData.RandomValues(rng, TextLength * config.CaptionChannels), TextLength, TextLength);
        LongCatPrompt negative = new(LongCatTestData.RandomValues(rng, TextLength * config.CaptionChannels), TextLength, TextLength);
        LongCatAvatarRun run = new(config.InChannels, Frames, Height, Width, LongCatTestData.RandomValues(rng, config.InChannels * Frames * Hw),
            positive, negative, LongCatTestData.RandomValues(rng, Frames * ctx * width), LongCatTestData.RandomValues(rng, Frames * ctx * width),
            steps, LongCatAvatarSchedule.Kind.Linear, 7f, 2f, 1.5f, LongCatAvatarMode.Avc, cond,
            LongCatTestData.RandomValues(rng, config.InChannels * cond * Hw), UseKvCache: true);
        cuda.ResetD2hSyncCount();
        new LongCatAvatarPipeline(cuda, transformer).Denoise(run);
        return cuda.GetD2hSyncCount();
    }

    private static string PtxDirectory()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate))
                return candidate;
        }

        throw new DirectoryNotFoundException("HartsyInference.Cuda/Ptx was not found above the test binary.");
    }
}
