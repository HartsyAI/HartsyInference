using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Every shader that declares a device feature beyond the Vulkan 1.3 baseline must be known to the kernel registry, so a device without that feature is refused by name before pipeline creation. The expectation is read from the shaders' own <c>#extension … : require</c> lines, so a new shader that adds one cannot slip past unmapped — and a mapping that over-reaches (a name prefix catching a shader with no such requirement) fails the same way.</summary>
public sealed class VulkanShaderFeatureLintTests
{
    private static readonly (string Extension, string Feature)[] Gated =
    [
        ("GL_EXT_shader_explicit_arithmetic_types_int64", "shaderInt64"),
        ("GL_EXT_shader_explicit_arithmetic_types_int16", "shaderInt16"),
    ];

    [Fact]
    public void EveryGatedExtensionMapsToItsFeature()
    {
        string shadersDir = FindShadersDir();
        int examined = 0;
        foreach (string path in Directory.EnumerateFiles(shadersDir, "*.comp.glsl"))
        {
            string name = Path.GetFileName(path)[..^".comp.glsl".Length];
            string text = File.ReadAllText(path);
            string? declared = null;
            foreach ((string extension, string feature) in Gated)
            {
                if (text.Contains($"#extension {extension} : require", StringComparison.Ordinal)) declared = feature;
            }
            string? mapped = VulkanKernelRegistry.RequiredFeature(name);
            Assert.True(declared == mapped,
                $"{name}: the shader declares {declared ?? "no gated feature"}, the registry maps {mapped ?? "none"}");
            examined++;
        }
        Assert.True(examined > 50, $"only {examined} shaders under {shadersDir}; the lint is not looking at the real set");
    }

    [Theory]
    [InlineData("im2col_f32", "shaderInt64")]
    [InlineData("im2col_f16", "shaderInt64")]
    [InlineData("cast_bf16_f32", "shaderInt16")]
    [InlineData("matmul_tiled_f16", null)]
    public void DispatchNamesResolveThroughTheirBaseShader(string dispatchName, string? feature)
    {
        Assert.Equal(feature, VulkanKernelRegistry.RequiredFeature(dispatchName));
    }

    private static string FindShadersDir()
    {
        string? dir = AppContext.BaseDirectory;
        while (dir is not null)
        {
            string candidate = Path.Combine(dir, "src", "HartsyInference.Vulkan", "Shaders");
            if (Directory.Exists(candidate)) return candidate;
            dir = Path.GetDirectoryName(dir);
        }
        throw new DirectoryNotFoundException("src/HartsyInference.Vulkan/Shaders not found above the test binary");
    }
}
