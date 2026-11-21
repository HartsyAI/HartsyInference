using HartsyInference.Core.Configuration;
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
/// without the hold is run with this class alone in its process and <c>diagnostics.vkKeepInstance</c> off. In that
/// run the only failure there can be is the crash: the second case checks nothing with the setting off.</para>
/// <para>Bound written before the first run with the hold: every round leaves the process alive. Measured on an
/// RTX 3060 with driver 610.88, this class alone in its process: with the hold off the host crashed in 5 of 5 runs at
/// 120 rounds and 3 of 5 at 40; with it on, 0 of 5.</para></summary>
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
        const int rounds = 120;
        for (int round = 0; round < rounds; round++)
        {
            using VulkanBackend backend = new();
            using Tensor q = Filled(new TensorShape(1, 2, 6, 64), 1), k = Filled(new TensorShape(1, 2, 9, 64), 2), v = Filled(k.Shape, 3);
            using Tensor output = new(q.Shape, DType.F32);
            // Head dim 64 with the F16 permission is the cooperative-matrix kernel where the device has it: the call
            // after which destroying the backend ended the process.
            backend.ScaledDotProductAttention(output, q, k, v, null, 0.125f, allowF16: true);
            backend.Sync();
            // Reading the output back each round makes the crash rarer, so only the last round checks its values.
            if (round == rounds - 1) Assert.True(float.IsFinite(output.AsReadOnlySpan<float>()[0]), "The last output is not finite.");
        }
    }

    /// <summary>The managed half of the same guard: with the setting on, a process that has created a backend holds
    /// its idle instance, and still holds it after that backend is gone. The probe behind <see cref="VulkanFactAttribute"/>
    /// has usually created a backend already, so this is a statement about the process, not about the backend below.
    /// With the setting off there is nothing to hold and nothing is checked.</summary>
    [VulkanFact]
    public void AProcessThatCreatedABackendHoldsItsInstance()
    {
        bool keep = EngineKnobs.VkKeepInstance.Value;
        if (!keep) return;
        using (VulkanBackend backend = new())
        {
            Assert.True(VulkanInstance.HoldsProcessInstance,
                $"No instance is held while a backend is alive (diagnostics.vkKeepInstance is {keep}; a failed hold is logged as a warning).");
        }
        Assert.True(VulkanInstance.HoldsProcessInstance,
            $"The held instance went away with the backend (diagnostics.vkKeepInstance is {keep}).");
    }
}
