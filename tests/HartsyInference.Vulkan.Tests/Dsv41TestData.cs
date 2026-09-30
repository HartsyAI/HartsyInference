using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Tensor helpers and backend construction shared by the DeepSeek-V4.1 primitive parity tests.</summary>
internal static unsafe class Dsv41TestData
{
    /// <summary>The device the parity tests run on, or null with a reason when no Vulkan device can be opened.</summary>
    public static VulkanBackend? TryCreateBackend(out string? skipReason)
    {
        try
        {
            VulkanBackend backend = VulkanTestDevice.Create();
            // A shared machine can name the one card these tests may touch; a construction on any other is only an instance probe.
            string? required = Environment.GetEnvironmentVariable("HARTSY_TEST_VULKAN_REQUIRE_NAME");
            if (!string.IsNullOrEmpty(required) && !backend.Vk.DeviceName.Contains(required, StringComparison.OrdinalIgnoreCase))
            {
                string actual = backend.Vk.DeviceName;
                backend.Dispose();
                skipReason = $"device '{actual}' does not contain required name '{required}'";
                return null;
            }
            skipReason = null;
            return backend;
        }
        catch (Exception ex)
        {
            skipReason = ex.Message;
            return null;
        }
    }

    public static Tensor F32(float[] data, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < data.Length; i++) p[i] = data[i];
        return t;
    }

    public static Tensor I32(int[] data, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.I32);
        int* p = (int*)t.DataPointer;
        for (int i = 0; i < data.Length; i++) p[i] = data[i];
        return t;
    }

    public static Tensor EmptyF32(params long[] shape) => new(new TensorShape(shape), DType.F32);

    public static Tensor EmptyI32(params long[] shape) => new(new TensorShape(shape), DType.I32);

    public static float[] ReadF32(Tensor t)
    {
        float[] r = new float[t.ElementCount];
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < r.Length; i++) r[i] = p[i];
        return r;
    }

    public static int[] ReadI32(Tensor t)
    {
        int[] r = new int[t.ElementCount];
        int* p = (int*)t.DataPointer;
        for (int i = 0; i < r.Length; i++) r[i] = p[i];
        return r;
    }

    public static float[] Random(int count, int seed, float scale = 1f)
    {
        Random rng = new(seed);
        float[] r = new float[count];
        for (int i = 0; i < r.Length; i++) r[i] = ((float)rng.NextDouble() * 2f - 1f) * scale;
        return r;
    }

    public static float MaxAbsDiff(float[] a, float[] b)
    {
        Xunit.Assert.Equal(a.Length, b.Length);
        float m = 0f;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    /// <summary>Bit-for-bit equality; every NaN counts as equal since its sign and payload carry no information.</summary>
    public static void AssertBitsEqual(float[] cpu, float[] vk, string what)
    {
        Xunit.Assert.Equal(cpu.Length, vk.Length);
        for (int i = 0; i < cpu.Length; i++)
            Xunit.Assert.True((float.IsNaN(cpu[i]) && float.IsNaN(vk[i])) ||
                BitConverter.SingleToInt32Bits(cpu[i]) == BitConverter.SingleToInt32Bits(vk[i]),
                $"{what} element {i}: cpu {cpu[i]:R} vk {vk[i]:R}");
    }
}
