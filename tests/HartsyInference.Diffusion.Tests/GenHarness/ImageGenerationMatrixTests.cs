using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Tests.GenHarness;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The consolidated image-generation harness that replaces per-model gen boilerplate. It has three lanes:
/// <list type="bullet">
///   <item><b>Well-formedness</b> (<see cref="Manifest_IsWellFormed"/>) — a Unit test (no trait) that validates the
///   manifest itself with no GPU/checkpoint, so the harness is exercised on every PR.</item>
///   <item><b>Deterministic matrix</b> (<see cref="Generate_Model"/>) — one theory case per registered model; each
///   skips cleanly when its weights or a CUDA device are absent, else renders a small fixed-seed image and asserts it
///   is renderable. Runs on the GPU lane for every model that is downloaded.</item>
///   <item><b>Seeded fuzz</b> (<see cref="Generate_RandomModel_Fuzz"/>) — the nightly "pick a random model" lane. It
///   selects one available model with a seed that is <em>printed</em>, so a red night reproduces exactly by exporting
///   <c>HARTSY_GEN_FUZZ_SEED</c>. Deterministic by construction; never a mystery failure.</item>
/// </list>
/// The class is named <c>*Generation*</c> so the self-hosted GPU lane (which selects by <c>FullyQualifiedName~Generation</c>)
/// picks it up; the gen methods carry tier traits so the CPU Unit gate excludes them.</summary>
public sealed class ImageGenerationMatrixTests
{
    private const string FuzzGate = "HARTSY_GEN_FUZZ";
    private const string FuzzSeedVar = "HARTSY_GEN_FUZZ_SEED";
    private static readonly ImageGenRequest SmokeRequest = new()
    {
        Prompt = "a majestic lion in a field of sunflowers, golden hour, photorealistic",
        Width = 128,
        Height = 128,
        Steps = 3,
        Seed = 42,
    };

    private readonly ITestOutputHelper _output;
    public ImageGenerationMatrixTests(ITestOutputHelper output) => _output = output;

    public static IEnumerable<object[]> CaseNames =>
        ImageModelManifest.All.Select(c => new object[] { c.Name });

    [Fact]
    public void Manifest_IsWellFormed()
    {
        Assert.NotEmpty(ImageModelManifest.All);
        Assert.Equal(
            ImageModelManifest.All.Select(c => c.Name).Distinct(StringComparer.Ordinal).Count(),
            ImageModelManifest.All.Count);
        foreach (ImageGenCase c in ImageModelManifest.All)
        {
            Assert.False(string.IsNullOrWhiteSpace(c.Name));
            Assert.NotNull(c.IsAvailable);
            Assert.NotNull(c.Generate);
        }
    }

    [Theory]
    [Trait("Category", "Integration")]
    [MemberData(nameof(CaseNames))]
    public void Generate_Model(string name)
    {
        ImageGenCase model = ImageModelManifest.All.Single(c => c.Name == name);
        RenderAndAssert(model, SmokeRequest);
    }

    [Fact]
    [Trait("Category", "Slow")]
    public void Generate_RandomModel_Fuzz()
    {
        if (Environment.GetEnvironmentVariable(FuzzGate) != "1")
        {
            _output.WriteLine($"SKIPPED: set {FuzzGate}=1 to run the nightly random-model fuzz lane.");
            return;
        }

        int seed = ResolveFuzzSeed();
        _output.WriteLine($"Fuzz seed = {seed} (export {FuzzSeedVar}={seed} to reproduce this exact selection).");

        List<ImageGenCase> available = ImageModelManifest.All.Where(c => c.IsAvailable()).ToList();
        if (available.Count == 0)
        {
            _output.WriteLine("SKIPPED: no registered model has its weights present.");
            return;
        }

        Random rng = new(seed);
        ImageGenCase chosen = available[rng.Next(available.Count)];
        _output.WriteLine($"Fuzz selected model: {chosen.Name} (from {available.Count} available).");
        RenderAndAssert(chosen, SmokeRequest with { Seed = rng.Next() });
    }

    private void RenderAndAssert(ImageGenCase model, ImageGenRequest request)
    {
        if (!model.IsAvailable())
        {
            _output.WriteLine($"SKIPPED: {model.Name} weights/tokenizer not present.");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine($"SKIPPED: {model.Name} needs a CUDA device (CPU generation is too slow for CI).");
            return;
        }

        using IBackend backend = new CudaBackend();
        _output.WriteLine($"Generating {request.Width}x{request.Height} @ {request.Steps} steps for {model.Name} (seed {request.Seed})...");
        GenImage image = model.Generate(backend, request);
        _output.WriteLine($"  {model.Name}: rendered {image.Width}x{image.Height}, seed {image.Seed}.");
        ImageValidity.AssertRenderable(image, request);
    }

    private static int ResolveFuzzSeed()
    {
        string? fromEnv = Environment.GetEnvironmentVariable(FuzzSeedVar);
        if (!string.IsNullOrEmpty(fromEnv) && int.TryParse(fromEnv, out int parsed))
            return parsed;
        // No seed pinned: derive a fresh one so nightly runs vary, but it is printed for exact reproduction.
        return unchecked((int)DateTime.UtcNow.Ticks);
    }
}
