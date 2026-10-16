using HartsyInference.ModelAssets.Lora;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>MOSS-Image's adaptable modules: every linear layer of the transformer, named as OpenMOSS's diffusers export,
/// its ComfyUI single file and <see cref="MossImageCheckpointConverter"/>'s output all name them (they agree). A LoRA trained
/// on any of those spells a module the same way under its wrapper, or flattens it for kohya.</summary>
public static class MossImageKeyNames
{
    /// <summary>The linear modules, with <c>{i}</c> for the joint-block or text-block index.</summary>
    public static LoraModuleTemplates Modules { get; } = new LoraModuleTemplates(
    [
        "video_embedder",
        "text_embedder.proj",
        "text_embedder.blocks.{i}.qkv",
        "text_embedder.blocks.{i}.proj",
        "text_embedder.blocks.{i}.ffn_gate",
        "text_embedder.blocks.{i}.ffn_up",
        "text_embedder.blocks.{i}.ffn_down",
        "time_embedding.0",
        "time_embedding.2",
        "shared_block_modulation",
        "layers.{i}.proj.q",
        "layers.{i}.proj.k",
        "layers.{i}.proj.v",
        "layers.{i}.proj.o",
        "layers.{i}.proj.ffn_up_gate",
        "layers.{i}.proj.ffn_down",
        "video_head_modulation",
        "final_linear_video",
    ]);

    /// <summary>Whether <paramref name="module"/> exists only in MOSS-Image. Every module but the timestep MLP qualifies:
    /// Wan names its own <c>time_embedding.0</c>/<c>.2</c> the same way.</summary>
    public static bool IsDistinctive(string module)
    {
        ArgumentNullException.ThrowIfNull(module);
        return Modules.IsModule(module) && !module.StartsWith("time_embedding.", StringComparison.Ordinal);
    }
}
