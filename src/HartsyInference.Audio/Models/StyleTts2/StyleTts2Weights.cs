using System.Globalization;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.StyleTts2;

/// <summary>Adapts the published StyleTTS2-LibriTTS checkpoint (<c>yl4579/StyleTTS2-LibriTTS</c>
/// <c>epochs_2nd_00020.pth</c>) to the flat dotted keys the engine's submodules expect. The raw checkpoint is
/// <c>{net: {component: {module.…: tensor}}}</c> (a dict-of-state-dicts under a <c>nn.DataParallel</c> wrapper);
/// a recursive-flatten load yields <c>net.{component}.module.{inner}</c>. This maps each to
/// <c>{component}.{inner}</c> (Kokoro's <c>bert/text_encoder/predictor/decoder</c> loaders + <c>bert_encoder</c>
/// consume these directly), and remaps the two StarGAN-v2 style encoders' <c>shared.N</c> Sequential indices to
/// the engine's semantic <c>stem</c> / <c>blocks.0-3</c> / <c>tail</c> names.</summary>
public static class StyleTts2Weights
{
    /// <summary>Returns a new dict keyed as <c>{component}.{inner}</c> (with the StyleEncoder <c>shared.N</c>
    /// remap applied), from a recursively-flattened load of the LibriTTS <c>.pth</c>.</summary>
    public static Dictionary<string, Tensor> Adapt(IReadOnlyDictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> o = new(raw.Count, StringComparer.Ordinal);
        foreach ((string k, Tensor v) in raw)
        {
            string s = k;
            if (s.StartsWith("net.", StringComparison.Ordinal)) s = s[4..];
            // Strip the single DataParallel `.module.` wrapper (`{comp}.module.{inner}` → `{comp}.{inner}`).
            s = s.Replace(".module.", ".", StringComparison.Ordinal);
            s = RemapStyleEncoder(s);
            o[s] = v;
        }
        return o;
    }

    /// <summary>The StyleEncoder / PredictorEncoder store their conv stack as a single <c>shared</c> Sequential:
    /// index 0 = stem conv, 1-4 = the four downsampling ResBlks, 5 = LeakyReLU (no weights), 6 = the 5×5 tail
    /// conv. The engine names these <c>stem</c> / <c>blocks.0-3</c> / <c>tail</c>.</summary>
    private static string RemapStyleEncoder(string s)
    {
        foreach (string enc in _styleEncoders)
        {
            string sharedPrefix = enc + ".shared.";
            if (!s.StartsWith(sharedPrefix, StringComparison.Ordinal)) continue;
            string rest = s[sharedPrefix.Length..];               // e.g. "1.conv1.weight_orig"
            int dot = rest.IndexOf('.');
            string idxStr = dot < 0 ? rest : rest[..dot];
            string tail = dot < 0 ? "" : rest[dot..];             // ".conv1.weight_orig"
            if (!int.TryParse(idxStr, NumberStyles.Integer, CultureInfo.InvariantCulture, out int idx)) return s;
            string? mapped = idx switch
            {
                0 => "stem",
                >= 1 and <= 4 => $"blocks.{idx - 1}",
                6 => "tail",
                _ => null
            };
            return mapped is null ? s : $"{enc}.{mapped}{tail}";
        }
        return s;
    }

    private static readonly string[] _styleEncoders = ["style_encoder", "predictor_encoder"];
}
