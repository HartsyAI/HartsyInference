namespace HartsyInference.ThreeD.Tests.Pixal3D;

/// <summary>A <see cref="FactAttribute"/> reported as <i>skipped</i> (never passed) unless the Pixal3D flow checkpoints are installed (as for
/// <see cref="Pixal3DRealWeightsFactAttribute"/>) and the reference dump folder <c>dumpFolder</c> exists under the folder named by
/// <c>PIXAL3D_FLOW_DUMPS</c> holding <c>dumpFile</c>.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class Pixal3DFlowParityFactAttribute : FactAttribute
{
    public Pixal3DFlowParityFactAttribute(string dumpFolder, string dumpFile, params string[] requiredFiles)
    {
        string? folder = Pixal3DRealWeightsFactAttribute.FindFolder();
        string? dumps = Environment.GetEnvironmentVariable("PIXAL3D_FLOW_DUMPS");
        if (folder is null)
        {
            Skip = "SKIPPED: Models/3D/pixal3d not found; the flow parity check did not run.";
        }
        else if (requiredFiles.FirstOrDefault(f => !File.Exists(Path.Combine(folder, f))) is { } missing)
        {
            Skip = $"SKIPPED: Models/3D/pixal3d/{missing} not found; the flow parity check did not run.";
        }
        else if (string.IsNullOrWhiteSpace(dumps) || !File.Exists(Path.Combine(dumps, dumpFolder, dumpFile)))
        {
            Skip = $"SKIPPED: PIXAL3D_FLOW_DUMPS does not name a folder with '{dumpFolder}/{dumpFile}' (see tests/python-reference/dump_pixal3d_*_flow*.py); the flow parity check did not run.";
        }
    }
}
