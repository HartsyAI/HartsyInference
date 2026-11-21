namespace HartsyInference.Vulkan;

/// <summary>Which dense F32 attention calls take the query-tiled kernel (<c>numerics.vkTiledAttention</c>).</summary>
public enum VulkanTiledAttention
{
    /// <summary>None: the scalar kernel up to its head dim and the materialized row above it.</summary>
    Off,

    /// <summary>Every call up to the scalar kernel's head dim; above it, only a call the materialized row cannot tile.</summary>
    Auto,

    /// <summary>Every call the kernel serves, at every head dim.</summary>
    All,
}
