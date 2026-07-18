using System.Text;

namespace HartsyInference.Engine;

/// <summary>Turns free text (a prompt) into a short, filesystem-safe slug for artifact names.</summary>
public static class Slug
{
    /// <summary>Lower-cases, keeps alphanumerics, collapses runs of separators to single hyphens, caps at 30 chars.</summary>
    public static string Make(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return "output";
        StringBuilder sb = new StringBuilder(32);
        foreach (char c in text.Trim())
        {
            if (sb.Length >= 30)
                break;
            if (char.IsLetterOrDigit(c))
                sb.Append(char.ToLowerInvariant(c));
            else if (c is ' ' or '-' or '_' && sb.Length > 0 && sb[^1] != '-')
                sb.Append('-');
        }
        string slug = sb.ToString().Trim('-');
        return slug.Length == 0 ? "output" : slug;
    }
}
