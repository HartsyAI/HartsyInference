using HartsyInference.ThreeD.Pipelines;

namespace HartsyInference.ThreeD.Tests.UniRig;

/// <summary>A <see cref="FactAttribute"/> reported as <i>skipped</i> unless <c>HARTSY_UNIRIG_DIR</c> names a folder holding both UniRig
/// checkpoints. xunit 2.x has no runtime skip, so the skip is set at discovery.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class UniRigRealWeightsFactAttribute : FactAttribute
{
    public UniRigRealWeightsFactAttribute()
    {
        string? root = Environment.GetEnvironmentVariable("HARTSY_UNIRIG_DIR");
        foreach (string relative in new[] { UniRigPipeline.SkeletonCheckpoint, UniRigPipeline.SkinCheckpoint })
        {
            if (string.IsNullOrWhiteSpace(root) || !File.Exists(Path.Combine(root, relative)))
            {
                Skip = $"SKIPPED: HARTSY_UNIRIG_DIR is unset or lacks {relative}; the UniRig real-weight run did not happen.";
                return;
            }
        }
    }
}
