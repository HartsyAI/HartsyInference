using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>A process may create a Vulkan backend, run attention on it and destroy it, over and over. On the NVIDIA
/// Windows driver that ended the process: a helper thread of the driver started on memory the driver had released
/// while the backend's device was being destroyed, and the test host died with an access violation in
/// <c>nvoglv64.dll</c> and no managed exception. <see cref="VulkanInstance"/> now holds one instance for the life of the
/// process, which keeps the driver off that path.
/// <para>A regression here is a crashed test host, not a failed assertion: the run reports "Test host process crashed"
/// and exits 1. Any other Vulkan instance alive in the process hides it, so the check that the loop still crashes
/// without the hold is run with this class alone in its process and <c>diagnostics.vkKeepInstance</c> off.</para>
/// <para>Bound written before the first run with the hold: 40 rounds leave the process alive. Without it the same loop
/// ended the process within 15 rounds in 6 of 6 processes (RTX 3060, driver 610.88).</para></summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanBackendTeardownTests
{
    private static Tensor Filled(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        Random random = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = random.NextSingle() * 2f - 1f;
        return tensor;
    }

    [VulkanFact]
    public void BackendsCreatedUsedAndDestroyedInTurnLeaveTheProcessAlive()
    {
        const int rounds = 40;
        using Tensor q = Filled(new TensorShape(1, 2, 6, 64), 1), k = Filled(new TensorShape(1, 2, 9, 64), 2), v = Filled(k.Shape, 3);
        for (int round = 0; round < rounds; round++)
        {
            using VulkanBackend backend = new();
            using Tensor output = new(q.Shape, DType.F32);
            // Head dim 64 with the F16 permission is the cooperative-matrix kernel where the device has it: the call
            // after which destroying the backend ended the process.
            backend.ScaledDotProductAttention(output, q, k, v, null, 0.125f, allowF16: true);
            backend.Sync();
            Assert.True(float.IsFinite(output.AsReadOnlySpan<float>()[0]), $"round {round}: the output is not finite.");
        }
    }

    /// <summary>The managed half of the same guard: with the setting at its default, a process that has created a
    /// backend holds its idle instance, and still holds it after that backend is gone.</summary>
    [VulkanFact]
    public void AProcessThatCreatedABackendHoldsItsInstance()
    {
        using (VulkanBackend backend = new())
        {
            Assert.True(VulkanInstance.HoldsProcessInstance, "No instance is held while a backend is alive.");
        }
        Assert.True(VulkanInstance.HoldsProcessInstance, "The held instance went away with the backend.");
    }
}
