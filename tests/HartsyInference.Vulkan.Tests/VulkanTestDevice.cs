namespace HartsyInference.Vulkan.Tests;

/// <summary>The device the Vulkan tests construct their backend on: <c>HARTSY_TEST_VULKAN_DEVICE</c> names an ordinal (a
/// second card while the first is busy with a gate), otherwise the backend's own default. A test-harness switch, not an engine knob.</summary>
internal static class VulkanTestDevice
{
    public static VulkanBackend Create() =>
        int.TryParse(Environment.GetEnvironmentVariable("HARTSY_TEST_VULKAN_DEVICE"), out int ordinal) ? new VulkanBackend(ordinal) : new VulkanBackend();
}
