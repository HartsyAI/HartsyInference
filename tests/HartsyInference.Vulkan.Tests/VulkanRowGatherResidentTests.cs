using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>The device row gather behind a prefill's last-row slice. A device-resident input is sliced on the device and must
/// match the host values exactly. A host-only input, or a row index outside the input, is refused so the caller takes the
/// host path, and the output is left untouched.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanRowGatherResidentTests(ITestOutputHelper output)
{
    private static bool VulkanAvailable()
    {
        try { using VulkanInstance i = new(); return i.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    [Theory]
    [InlineData(1024, 5, new[] { 4 })]           // the last-row slice of a 5-token prefill
    [InlineData(64, 7, new[] { 0, 6, 3 })]       // several rows, in a different order
    public void ResidentInput_GathersTheRowsOnTheDevice(int width, int rows, int[] pick)
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        using Tensor x = new(new TensorShape(rows, width), DType.F32);
        Span<float> xs = x.AsSpan<float>();
        for (int i = 0; i < xs.Length; i++) xs[i] = ((i * 37) % 101) * 0.05f - 2.5f;
        using Tensor y = new(new TensorShape(rows, width), DType.F32);
        backend.Silu(y, x);   // y now lives on the device as a cached activation
        using Tensor gathered = new(new TensorShape(pick.Length, width), DType.F32);
        Assert.True(backend.TryGatherRowsResident(gathered, y, pick));
        float[] all = y.AsReadOnlySpan<float>().ToArray();
        ReadOnlySpan<float> got = gathered.AsReadOnlySpan<float>();
        for (int m = 0; m < pick.Length; m++)
            for (int c = 0; c < width; c++)
                Assert.Equal(all[pick[m] * width + c], got[m * width + c]);
    }

    [Fact]
    public void HostOnlyInput_IsRefused_AndTheOutputIsUntouched()
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        using Tensor host = new(new TensorShape(4, 8), DType.F32);
        host.AsSpan<float>().Fill(3f);
        using Tensor gathered = new(new TensorShape(1, 8), DType.F32);
        gathered.AsSpan<float>().Fill(-7f);
        Assert.False(backend.TryGatherRowsResident(gathered, host, [1]));
        Assert.All(gathered.AsReadOnlySpan<float>().ToArray(), v => Assert.Equal(-7f, v));
    }

    [Fact]
    public void RowOutsideTheInput_IsRefused()
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        using Tensor x = new(new TensorShape(3, 16), DType.F32);
        x.AsSpan<float>().Fill(1f);
        using Tensor y = new(new TensorShape(3, 16), DType.F32);
        backend.Silu(y, x);
        using Tensor gathered = new(new TensorShape(1, 16), DType.F32);
        Assert.False(backend.TryGatherRowsResident(gathered, y, [3]));
        Assert.False(backend.TryGatherRowsResident(gathered, y, [-1]));
    }
}
