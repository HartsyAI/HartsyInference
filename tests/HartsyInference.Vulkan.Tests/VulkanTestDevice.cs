using HartsyInference.Tests.Common;

namespace HartsyInference.Vulkan.Tests;

/// <summary>The device the Vulkan tests construct: <c>HARTSY_TEST_VULKAN_DEVICE</c> names an ordinal (a second card
/// while the first is busy with a gate), otherwise the backend's own default. A test-harness switch, not an engine knob.
/// The parsing and its failure modes live in <see cref="VulkanTestDeviceSelection"/>, which <c>BackendGate</c> shares.</summary>
internal static class VulkanTestDevice
{
    /// <summary>The backend for a test: the pinned ordinal when the variable is set, the default picker otherwise.
    /// <paramref name="spvDir"/> overrides the SPIR-V directory exactly as the backend's own constructor does.</summary>
    public static VulkanBackend Create(string? spvDir = null) =>
        new VulkanBackend(VulkanTestDeviceSelection.FromEnvironment(), spvDir);

    /// <summary>The logical device for a test that drives <see cref="VulkanDevice"/> itself, under the same pin as <see cref="Create"/>.</summary>
    public static VulkanDevice CreateDevice(VulkanInstance instance) =>
        VulkanDevice.Create(instance, VulkanTestDeviceSelection.FromEnvironment());
}
