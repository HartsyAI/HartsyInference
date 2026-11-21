using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Tests.Parity;
using HartsyInference.Vulkan;

namespace HartsyInference.LLM.Tests;

/// <summary>A real driver's pipeline-cache blob through the file checks, in a directory this test owns.
/// <para>Bounds fixed before the first run: the first backend starts with no initial data; the file it leaves
/// passes the checks against this device's vendor, device and pipeline-cache UUID; the second backend hands the
/// driver exactly that blob; a third, started on the file cut to six tenths, hands the driver nothing, still
/// computes, and leaves a whole file again.</para></summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanPipelineCachePersistTests
{
    [VulkanFact]
    public void CacheFilePersistsReloadsAndRecoversFromACutShortFile()
    {
        string dir = Path.Combine(Path.GetTempPath(), "hartsy-vkpc-gpu-" + Guid.NewGuid().ToString("N"));
        try
        {
            string path;
            VulkanPipelineCacheIdentity identity;
            using (VulkanBackend first = new(null, null, dir))
            {
                path = first.PipelineCachePath;
                identity = new(first.Vk.VendorId, first.Vk.DeviceId, first.Vk.PipelineCacheUuid);
                Assert.Equal(dir, Path.GetDirectoryName(path));
                Assert.Equal(0, first.PipelineCacheInitialDataBytes);
                RunOneKernel(first);
            }

            VulkanPipelineCacheLoad written = VulkanPipelineCacheFile.Load(path, identity);
            Assert.Equal(VulkanPipelineCacheLoadStatus.Loaded, written.Status);
            Assert.True(written.Blob.Length >= VulkanPipelineCacheFile.DriverHeaderSize);

            using (VulkanBackend second = new(null, null, dir))
            {
                Assert.Equal(written.Blob.Length, second.PipelineCacheInitialDataBytes);
                RunOneKernel(second);
            }

            byte[] file = File.ReadAllBytes(path);
            File.WriteAllBytes(path, file[..(file.Length * 6 / 10)]);
            using (VulkanBackend third = new(null, null, dir))
            {
                Assert.Equal(0, third.PipelineCacheInitialDataBytes);
                RunOneKernel(third);
            }
            Assert.Equal(VulkanPipelineCacheLoadStatus.Loaded, VulkanPipelineCacheFile.Load(path, identity).Status);
        }
        finally
        {
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }

    private static void RunOneKernel(VulkanBackend backend)
    {
        using Tensor input = new(new TensorShape(4, 8), DType.F32);
        Span<float> values = input.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = i - 16;
        using Tensor output = new(input.Shape, DType.F32);

        backend.LeakyRelu(output, input, 0.5f);

        ReadOnlySpan<float> got = output.AsReadOnlySpan<float>();
        for (int i = 0; i < got.Length; i++) Assert.Equal(i < 16 ? (i - 16) * 0.5f : i - 16, got[i]);
    }
}
