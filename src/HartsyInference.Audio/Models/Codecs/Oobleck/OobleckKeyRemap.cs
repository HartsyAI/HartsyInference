using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Oobleck;

/// <summary>Renames real diffusers <c>AutoencoderOobleck</c> named-submodule keys to the flat <c>nn.Sequential</c> numeric-index layout <see cref="OobleckEncoder"/>/<see cref="OobleckDecoder"/> expect.</summary>
/// <remarks>Verified against the released Stable Audio Open Small <c>vae/diffusion_pytorch_model.safetensors</c>:
/// <c>encoder.conv1</c>, <c>encoder.block.{i}.res_unit1.snake1</c>, <c>decoder.block.{i}.conv_t1</c>, ... map to
/// the dialect ACE-Step 1.5's checkpoint ships as — <c>encoder.layers.0</c>, <c>encoder.layers.{i+1}.layers.{j}</c>,
/// ... Pure key rename, same <see cref="Tensor"/> references, no data copy. Already-flat checkpoints and
/// unrelated keys pass through unchanged, so it is safe to call unconditionally before
/// <see cref="OobleckVae.LoadWeights"/>.</remarks>
public static class OobleckKeyRemap
{
    public static Dictionary<string, Tensor> ToFlatSequentialLayout(IReadOnlyDictionary<string, Tensor> named, OobleckConfig cfg)
    {
        int stages = cfg.DownsamplingRatios.Length;
        Dictionary<string, Tensor> flat = new(named.Count);
        foreach ((string key, Tensor value) in named)
            flat[MapKey(key, stages) ?? key] = value;
        return flat;
    }

    private static string? MapKey(string key, int stages)
    {
        string[] seg = key.Split('.');
        if (seg.Length < 3 || (seg[0] != "encoder" && seg[0] != "decoder"))
            return null;
        string side = seg[0];
        string tail = string.Join('.', seg[2..]);

        return seg[1] switch
        {
            "conv1" => $"{side}.layers.0.{tail}",
            "conv2" => $"{side}.layers.{stages + 2}.{tail}",
            "snake1" => $"{side}.layers.{stages + 1}.{tail}",
            "block" => MapBlockKey(side, seg, stages),
            _ => null,
        };
    }

    private static string? MapBlockKey(string side, string[] seg, int stages)
    {
        // seg = [side, "block", "<i>", "<component>", ...tail]
        if (seg.Length < 4 || !int.TryParse(seg[2], out int i) || i < 0 || i >= stages)
            return null;
        string blockPrefix = $"{side}.layers.{i + 1}";
        string component = seg[3];
        string tail = string.Join('.', seg[4..]);

        return side == "encoder"
            ? component switch
            {
                "res_unit1" => MapResidualUnit(blockPrefix, 0, tail),
                "res_unit2" => MapResidualUnit(blockPrefix, 1, tail),
                "res_unit3" => MapResidualUnit(blockPrefix, 2, tail),
                "snake1" => $"{blockPrefix}.layers.3.{tail}",
                "conv1" => $"{blockPrefix}.layers.4.{tail}",
                _ => null,
            }
            : component switch
            {
                "snake1" => $"{blockPrefix}.layers.0.{tail}",
                "conv_t1" => $"{blockPrefix}.layers.1.{tail}",
                "res_unit1" => MapResidualUnit(blockPrefix, 2, tail),
                "res_unit2" => MapResidualUnit(blockPrefix, 3, tail),
                "res_unit3" => MapResidualUnit(blockPrefix, 4, tail),
                _ => null,
            };
    }

    /// <summary>A residual unit's own flat layout: <c>layers.0</c>=snake1, <c>layers.1</c>=conv1, <c>layers.2</c>=snake2, <c>layers.3</c>=conv2 — see <see cref="OobleckResidualUnit"/>.</summary>
    private static string? MapResidualUnit(string blockPrefix, int slot, string tail)
    {
        string ruPrefix = $"{blockPrefix}.layers.{slot}";
        int dot = tail.IndexOf('.');
        if (dot < 0) return null;
        string sub = tail[..dot], rest = tail[(dot + 1)..];
        return sub switch
        {
            "snake1" => $"{ruPrefix}.layers.0.{rest}",
            "conv1" => $"{ruPrefix}.layers.1.{rest}",
            "snake2" => $"{ruPrefix}.layers.2.{rest}",
            "conv2" => $"{ruPrefix}.layers.3.{rest}",
            _ => null,
        };
    }
}
