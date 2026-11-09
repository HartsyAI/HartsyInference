namespace HartsyInference.LLM.Tests;

/// <summary>Finds the CUDA PTX kernels: beside the test assembly when copied there, else in the repository's
/// <c>src/HartsyInference.Cuda/Ptx</c> folder (the layout the CUDA test project uses).</summary>
internal static class CudaPtxDirectory
{
    public static string Resolve()
    {
        string beside = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (Directory.Exists(beside)) return beside;
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        throw new DirectoryNotFoundException("No CUDA PTX directory beside the test assembly or in the repository.");
    }
}
