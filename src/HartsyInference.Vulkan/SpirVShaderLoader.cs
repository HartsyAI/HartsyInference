using System.Runtime.InteropServices;

namespace HartsyInference.Vulkan;

/// <summary>Loads .spv files from disk and constructs <c>VkShaderModule</c> handles.</summary>
public static class SpirVShaderLoader
{
    /// <summary>Loads a SPIR-V binary from disk and creates a <c>VkShaderModule</c>. Caller takes ownership and must call <c>vkDestroyShaderModule</c>.</summary>
    public static unsafe ulong LoadModuleFromFile(nint device, string spvPath)
    {
        if (!File.Exists(spvPath))
            throw new FileNotFoundException($"SPIR-V file not found: {spvPath}", spvPath);

        byte[] bytes = File.ReadAllBytes(spvPath);
        if (bytes.Length == 0 || (bytes.Length & 3) != 0)
            throw new VulkanException(-200, $"SPIR-V file {spvPath} has invalid size {bytes.Length} (must be > 0 and multiple of 4)");
        return LoadModuleFromBytes(device, bytes);
    }

    public static unsafe ulong LoadModuleFromBytes(nint device, byte[] bytes)
    {
        nint pin = Marshal.AllocHGlobal(bytes.Length);
        try
        {
            Marshal.Copy(bytes, 0, pin, bytes.Length);
            VkShaderModuleCreateInfo ci = new()
            {
                sType = VkStructureType.ShaderModuleCreateInfo,
                codeSize = (nuint)bytes.Length,
                pCode = pin,
            };
            VulkanApi.vkCreateShaderModule(device, in ci, 0, out ulong module).ThrowOnError("vkCreateShaderModule");
            return module;
        }
        finally
        {
            Marshal.FreeHGlobal(pin);
        }
    }
}
