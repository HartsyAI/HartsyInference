using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.ModelHandler.CheckpointConverters;

/// <summary>Loads Matrix-Game 2.0 DiT checkpoints (<c>Skywork/Matrix-Game-2.0</c> — <c>base_distill</c> /
/// <c>gta_keyboard2dim</c> / <c>templerun_7dim_onlykey</c> / foundation safetensors) for
/// <c>MatrixGame2Transformer</c>. The key routing is <b>identical</b> to Matrix-Game 3.0 (original Wan naming +
/// per-block <c>action_model</c> weights + the I2V <c>img_emb</c> projection, all covered by the same verbatim
/// rename table), so this delegates to <see cref="MatrixGame3CheckpointConverter"/> and adds the MG2-specific shape
/// helpers (keyboard vocab inferred from <c>keyboard_embed.0.weight</c>, mouse presence from <c>mouse_mlp</c>).
/// The Wan2.1 VAE (<c>Wan2.1_VAE.pth</c>) must be converted to safetensors offline — its keys load 1:1 via
/// <see cref="LanceCheckpointConverter.LoadVae"/>'s prefix-stripping path.</summary>
public sealed class MatrixGame2CheckpointConverter
{
    /// <summary>Per-variant action shape read from the weights.</summary>
    public readonly record struct InferredActionShape(int KeyboardDim, bool HasMouse, int[] ActionBlocks);

    /// <summary>Converts a flat weight dictionary (same routing as Matrix-Game 3.0).</summary>
    public static MatrixGame3CheckpointConverter.ConvertedWeights Convert(Dictionary<string, Tensor> allWeights) =>
        MatrixGame3CheckpointConverter.Convert(allWeights);

    /// <summary>Loads a single safetensors file and converts. The caller owns the loader.</summary>
    public static (MatrixGame3CheckpointConverter.ConvertedWeights Weights, SafeTensorsLoader Loader) LoadAndConvert(string checkpointPath) =>
        MatrixGame3CheckpointConverter.LoadAndConvert(checkpointPath);

    /// <summary>Reads the per-variant action shape from converted weights: the keyboard one-hot width from the first
    /// <c>keyboard_embed.0.weight</c> ([128, K]) and mouse presence from any <c>mouse_mlp</c> key.</summary>
    public static InferredActionShape InferActionShape(MatrixGame3CheckpointConverter.ConvertedWeights converted)
    {
        int keyboardDim = 0;
        bool hasMouse = false;
        foreach ((string key, Tensor value) in converted.Transformer)
        {
            if (keyboardDim == 0 && key.EndsWith(".action_model.keyboard_embed.0.weight", StringComparison.Ordinal))
                keyboardDim = (int)value.Shape[1];
            if (!hasMouse && key.Contains(".action_model.mouse_mlp.", StringComparison.Ordinal))
                hasMouse = true;
            if (keyboardDim != 0 && hasMouse) break;
        }
        return new InferredActionShape(keyboardDim, hasMouse, converted.ActionBlocks);
    }
}
