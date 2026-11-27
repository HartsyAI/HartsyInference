namespace HartsyInference.LLM.Tests.Glm5Next;

/// <summary>The key set of a published <c>model.safetensors.index.json</c>, stored as <c>pattern TAB layers TAB experts</c> lines (<c>{l}</c> a
/// layer or block, <c>{e}</c> an expert, ranges like <c>3-44,45</c>) so the repository carries no weights and no 3 MB index.</summary>
internal static class Glm5NextIndexFixture
{
    internal static string Path(string name) => System.IO.Path.Combine(AppContext.BaseDirectory, "Fixtures", "Glm5Next", name);

    internal static string ConfigJson() => File.ReadAllText(Path("sage-s21-flash-config.json")).Replace("\r\n", "\n");

    internal static HashSet<string> Load(string name)
    {
        HashSet<string> keys = new(StringComparer.Ordinal);
        foreach (string line in File.ReadLines(Path(name)))
        {
            if (line.Length == 0) continue;
            string[] f = line.Split('\t');
            int experts = f[2] == "-" ? 0 : int.Parse(f[2]);
            IEnumerable<int> layers = f[1] == "-" ? [-1] : Ranges(f[1]);
            foreach (int l in layers)
            {
                string p = l < 0 ? f[0] : f[0].Replace("{l}", l.ToString());
                if (experts == 0) keys.Add(p);
                else for (int e = 0; e < experts; e++) keys.Add(p.Replace("{e}", e.ToString()));
            }
        }
        return keys;
    }

    private static IEnumerable<int> Ranges(string spec)
    {
        foreach (string part in spec.Split(','))
        {
            string[] ab = part.Split('-');
            int a = int.Parse(ab[0]), b = ab.Length == 2 ? int.Parse(ab[1]) : a;
            for (int i = a; i <= b; i++) yield return i;
        }
    }
}
