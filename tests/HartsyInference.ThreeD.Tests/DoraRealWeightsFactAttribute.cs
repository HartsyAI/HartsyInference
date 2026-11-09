namespace HartsyInference.ThreeD.Tests;

/// <summary>A <see cref="FactAttribute"/> reported as <i>skipped</i> (never passed) when the Dora real-weight inputs are
/// missing: <c>DORA_CKPT</c> (the released <c>dora_vae_1_1.ckpt</c>) and <c>DORA_REFERENCE</c> (the folder written by
/// <c>tools/parity/dump_dora_encoder.py</c>). With <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> it runs and fails instead.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class DoraRealWeightsFactAttribute : FactAttribute
{
    public DoraRealWeightsFactAttribute()
    {
        if (Environment.GetEnvironmentVariable("HARTSY_REQUIRE_REAL_WEIGHTS") == "1")
            return;
        string? ckpt = Environment.GetEnvironmentVariable("DORA_CKPT");
        string? reference = Environment.GetEnvironmentVariable("DORA_REFERENCE");
        if (ckpt is null || !File.Exists(ckpt))
            Skip = "SKIPPED: DORA_CKPT is not set to dora_vae_1_1.ckpt; Dora parity did not run.";
        else if (reference is null || !File.Exists(Path.Combine(reference, "manifest.json")))
            Skip = "SKIPPED: DORA_REFERENCE is not set to a dump_dora_encoder.py output; Dora parity did not run.";
    }
}
