using System.Text;

namespace HartsyInference.Diffusion.Prompting.Dialects;

/// <summary>Flattens a <see cref="StructuredPrompt"/> to a single prose string for plain-text models (SD/SDXL/Flux/Lens). Bbox placement is dropped; rendered-text elements are quoted. Lets a caller author one structured prompt and target any model.</summary>
public sealed class NaturalLanguageDialect : IPromptDialect
{
    /// <inheritdoc/>
    public string Name => "natural-language";

    /// <inheritdoc/>
    public string Serialize(StructuredPrompt prompt)
    {
        StringBuilder sb = new();
        if (!string.IsNullOrEmpty(prompt.Summary))
            sb.Append(prompt.Summary!.TrimEnd('.')).Append(". ");

        if (prompt.Style is { } style)
        {
            string medium = style.Photo ?? style.ArtStyle ?? style.Medium ?? "";
            AppendClause(sb, medium);
            AppendClause(sb, style.Lighting);
            AppendClause(sb, style.Aesthetics);
        }

        if (!string.IsNullOrEmpty(prompt.Background))
            sb.Append("Background: ").Append(prompt.Background!.TrimEnd('.')).Append(". ");

        foreach (PromptElement element in prompt.Elements)
        {
            if (element is TextElement t)
            {
                sb.Append("Text reading \"").Append(t.Text).Append('"');
                if (!string.IsNullOrEmpty(t.Description)) sb.Append(" (").Append(t.Description).Append(')');
                sb.Append(". ");
            }
            else if (!string.IsNullOrEmpty(element.Description))
            {
                sb.Append(element.Description!.TrimEnd('.')).Append(". ");
            }
        }

        return sb.ToString().TrimEnd();
    }

    private static void AppendClause(StringBuilder sb, string? clause)
    {
        if (!string.IsNullOrEmpty(clause))
            sb.Append(clause!.TrimEnd('.')).Append(". ");
    }
}
