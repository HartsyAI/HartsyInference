using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpInference.Vulkan;

/// <summary>Small helpers shared across the Vulkan backend: fixed-byte string conversion, version encoding, pinned UTF-8 string arrays for ppEnabledLayerNames / ppEnabledExtensionNames.</summary>
internal static class VulkanUtilities
{
    /// <summary>Encodes a Vulkan API version: VK_MAKE_VERSION(major, minor, patch).</summary>
    public static uint MakeVersion(uint major, uint minor, uint patch)
        => (major << 22) | (minor << 12) | patch;

    public const uint VK_API_VERSION_1_0 = (1u << 22) | (0u << 12) | 0u;
    public const uint VK_API_VERSION_1_1 = (1u << 22) | (1u << 12) | 0u;
    public const uint VK_API_VERSION_1_2 = (1u << 22) | (2u << 12) | 0u;
    public const uint VK_API_VERSION_1_3 = (1u << 22) | (3u << 12) | 0u;

    /// <summary>Reads a NUL-terminated UTF-8 string from a fixed-size byte buffer, e.g. VkPhysicalDeviceProperties.deviceName[256].</summary>
    public static unsafe string ReadFixedString(byte* buffer, int max)
    {
        int len = 0;
        while (len < max && buffer[len] != 0) len++;
        return Encoding.UTF8.GetString(buffer, len);
    }
}

/// <summary>
/// Holds an array of UTF-8 NUL-terminated strings as a single contiguous block
/// of unmanaged memory, with a parallel array of <c>const char**</c> pointers
/// suitable for Vulkan's <c>ppEnabledLayerNames</c> / <c>ppEnabledExtensionNames</c>.
/// Disposable — frees both blocks.
/// </summary>
internal sealed unsafe class PinnedStringArray : IDisposable
{
    private nint _stringsBlock;
    private nint _pointersBlock;

    /// <summary>Pointer to the array of pointers (suitable for ppEnabled*).</summary>
    public nint Pointers => _pointersBlock;
    public uint Count { get; }

    public PinnedStringArray(IReadOnlyList<string> strings)
    {
        Count = (uint)strings.Count;
        if (strings.Count == 0)
        {
            _stringsBlock = 0;
            _pointersBlock = 0;
            return;
        }

        // Compute total UTF-8 bytes (with NUL terminators)
        int totalBytes = 0;
        for (int i = 0; i < strings.Count; i++)
            totalBytes += Encoding.UTF8.GetByteCount(strings[i]) + 1;

        _stringsBlock = Marshal.AllocHGlobal(totalBytes);
        _pointersBlock = Marshal.AllocHGlobal(strings.Count * sizeof(nint));

        byte* dst = (byte*)_stringsBlock;
        nint* ptrs = (nint*)_pointersBlock;

        for (int i = 0; i < strings.Count; i++)
        {
            ptrs[i] = (nint)dst;
            int written = Encoding.UTF8.GetBytes(strings[i], new Span<byte>(dst, totalBytes - (int)((byte*)dst - (byte*)_stringsBlock)));
            dst[written] = 0;
            dst += written + 1;
        }
    }

    public void Dispose()
    {
        if (_stringsBlock != 0) { Marshal.FreeHGlobal(_stringsBlock); _stringsBlock = 0; }
        if (_pointersBlock != 0) { Marshal.FreeHGlobal(_pointersBlock); _pointersBlock = 0; }
    }
}

/// <summary>Pin a single UTF-8 NUL-terminated string for passing as a <c>const char*</c> (e.g. <c>VkPipelineShaderStageCreateInfo.pName</c> = "main").</summary>
internal sealed class PinnedUtf8String : IDisposable
{
    private nint _block;
    public nint Pointer => _block;

    public PinnedUtf8String(string s)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(s);
        _block = Marshal.AllocHGlobal(bytes.Length + 1);
        unsafe
        {
            byte* dst = (byte*)_block;
            for (int i = 0; i < bytes.Length; i++) dst[i] = bytes[i];
            dst[bytes.Length] = 0;
        }
    }

    public void Dispose()
    {
        if (_block != 0) { Marshal.FreeHGlobal(_block); _block = 0; }
    }
}
