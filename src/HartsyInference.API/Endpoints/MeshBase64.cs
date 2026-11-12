namespace HartsyInference.API.Endpoints;

/// <summary>Decodes the base64 a mesh route takes, turning a malformed value into the 400 an <see cref="ArgumentException"/> maps to.</summary>
internal static class MeshBase64
{
    /// <summary>The bytes of <paramref name="text"/>, which may carry a <c>data:</c> URI prefix.</summary>
    public static byte[] Decode(string text)
    {
        int comma = text.StartsWith("data:", StringComparison.Ordinal) ? text.IndexOf(',') : -1;
        try
        {
            return Convert.FromBase64String(comma >= 0 ? text[(comma + 1)..] : text);
        }
        catch (FormatException ex)
        {
            throw new ArgumentException("Data is not valid base64.", ex);
        }
    }
}
