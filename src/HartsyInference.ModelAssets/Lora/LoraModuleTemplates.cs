using System.Text;
using System.Text.RegularExpressions;

namespace HartsyInference.ModelAssets.Lora;

/// <summary>A model's adaptable module paths as templates, <c>{i}</c> standing for a block index
/// (<c>layers.{i}.proj.q</c>), answering the two questions every LoRA mapper asks of a key's module: is this dotted path
/// one of the model's modules, and which dotted path does a kohya name (<c>lora_unet_</c> stripped, every dot flattened to
/// an underscore) spell? Kohya's flattening is ambiguous on its own (<c>proj_ffn_up_gate</c> could split anywhere); the
/// templates resolve it, so a family describes its modules once instead of hand-writing a parser.</summary>
public sealed class LoraModuleTemplates
{
    private const string IndexToken = "{i}";

    private readonly (string Template, Regex Dotted, Regex Kohya)[] _patterns;

    /// <summary>Compiles <paramref name="templates"/>; order does not matter, since every pattern is anchored at both ends.</summary>
    public LoraModuleTemplates(IEnumerable<string> templates)
    {
        ArgumentNullException.ThrowIfNull(templates);
        Templates = [.. templates];
        _patterns = [.. Templates.Select(template => (template, Compile(template, '.'), Compile(template, '_')))];
    }

    /// <summary>The module templates, as given.</summary>
    public IReadOnlyList<string> Templates { get; }

    /// <summary>Whether <paramref name="module"/> (a dotted path without <c>.weight</c>) is one of the modules.</summary>
    public bool IsModule(string module)
    {
        ArgumentNullException.ThrowIfNull(module);
        foreach ((string _, Regex dotted, Regex _) in _patterns)
        {
            if (dotted.IsMatch(module))
                return true;
        }
        return false;
    }

    /// <summary>The template <paramref name="module"/> instantiates, or null when it is not one of the modules.</summary>
    public string? TemplateOf(string module)
    {
        ArgumentNullException.ThrowIfNull(module);
        foreach ((string template, Regex dotted, Regex _) in _patterns)
        {
            if (dotted.IsMatch(module))
                return template;
        }
        return null;
    }

    /// <summary>The dotted module a kohya name spells, or null when it spells none of them.</summary>
    public string? FromKohya(string flat)
    {
        ArgumentNullException.ThrowIfNull(flat);
        foreach ((string template, Regex _, Regex kohya) in _patterns)
        {
            Match match = kohya.Match(flat);
            if (!match.Success)
                continue;
            string[] parts = template.Split(IndexToken);
            StringBuilder module = new StringBuilder(parts[0]);
            for (int i = 1; i < parts.Length; i++)
                module.Append(match.Groups[i].Value).Append(parts[i]);
            return module.ToString();
        }
        return null;
    }

    /// <summary>An anchored pattern for <paramref name="template"/> with dots spelled <paramref name="separator"/> and each
    /// index captured as a decimal number.</summary>
    private static Regex Compile(string template, char separator)
    {
        IEnumerable<string> literals = template.Split(IndexToken).Select(part => Regex.Escape(part.Replace('.', separator)));
        return new Regex("^" + string.Join(@"(\d+)", literals) + "$", RegexOptions.CultureInvariant | RegexOptions.Compiled);
    }
}
