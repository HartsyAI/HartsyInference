namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>Locates the Python reference dumps (read from <c>HARTSY_PARITY_DIR</c>, never copied into the repository) and the
/// real checkpoints the parity tests need.</summary>
internal static class ParityFixtures
{
    internal const string ParityDirVariable = "HARTSY_PARITY_DIR";
    internal const string RequireWeightsVariable = "HARTSY_REQUIRE_REAL_WEIGHTS";
    internal const string DecomposerFixture = "decomposer_free_q20_fp32.npz";
    internal const string ShapeVaeFixture = "shapevae_f16c32_fp32_trellis2.npz";

    private static readonly string[] CheckpointFiles =
    [
        Path.Combine("CoLa3D", "cola3d_free_q20.pt"),
        Path.Combine("TRELLIS2-ShapeVAE", "shape_enc_next_dc_f16c32_fp16.safetensors"),
        Path.Combine("TRELLIS2-ShapeVAE", "shape_dec_next_dc_f16c32_fp16.safetensors"),
        Path.Combine("Utonia", "utonia.pth"),
    ];

    /// <summary>Resolves the fixture directory and the Models root. Returns false with a reason when either is missing, so the
    /// caller reports SKIPPED. With <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> a missing resource throws instead.</summary>
    internal static bool TryResolve(string[] fixtureFiles, out string parityDir, out string modelRoot, out string reason)
    {
        string? problem = ResolvePaths(fixtureFiles, out parityDir, out modelRoot);
        if (problem is null)
        {
            reason = string.Empty;
            return true;
        }

        if (Environment.GetEnvironmentVariable(RequireWeightsVariable) == "1")
            throw new InvalidOperationException($"{RequireWeightsVariable}=1 but {problem}.");
        reason = problem;
        return false;
    }

    private static string? ResolvePaths(string[] fixtureFiles, out string parityDir, out string modelRoot)
    {
        parityDir = Environment.GetEnvironmentVariable(ParityDirVariable) ?? string.Empty;
        modelRoot = FindModelRoot() ?? string.Empty;
        if (string.IsNullOrWhiteSpace(parityDir))
            return $"{ParityDirVariable} is not set";
        if (!Directory.Exists(parityDir))
            return $"{ParityDirVariable} points at a missing directory: {parityDir}";
        foreach (string file in fixtureFiles)
        {
            if (!File.Exists(Path.Combine(parityDir, file)))
                return $"parity fixture {file} not found in {parityDir}";
        }

        if (modelRoot.Length == 0)
            return "CoLa3D, TRELLIS2-ShapeVAE and Utonia checkpoints not found under a Models folder above the test binary";
        return null;
    }

    // Walks up from the test binary to the first Models folder holding every checkpoint the parity tests read. A worktree
    // finds the main checkout's Models this way because the worktree sits below it.
    private static string? FindModelRoot()
    {
        DirectoryInfo? dir = new(AppContext.BaseDirectory);
        while (dir is not null)
        {
            string models = Path.Combine(dir.FullName, "Models");
            if (CheckpointFiles.All(file => File.Exists(Path.Combine(models, file))))
                return models;
            dir = dir.Parent;
        }

        return null;
    }
}
