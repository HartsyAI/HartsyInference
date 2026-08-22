using System.Text.RegularExpressions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Features;

/// <summary>Resolves textual-inversion embed markers (<c>\0swarmembed:NAME\0end</c>, the control form hosts rewrite <c>&lt;embed:name&gt;</c> into) for CLIP pipelines: loads each embedding's learned vectors via <see cref="TextualInversion.Load"/> and builds a token sequence where each embedding's N vectors occupy N sequential placeholder ids past the tokenizer vocab. The caller encodes those tokens with the per-hidden-size <see cref="Plan.InlineMap"/> so the engine substitutes the learned vectors at the placeholder positions. Returns null when the prompt carries no markers — callers keep their plain token path unchanged.</summary>
public static class EmbeddingResolver
{
    private static readonly Regex _markerRx = new Regex("\u0000swarmembed:(.*?)\u0000end", RegexOptions.Compiled, TimeSpan.FromSeconds(2));

    /// <summary>Models-root-relative folders searched for embedding files.</summary>
    private static readonly string[] _folders = ["Embeddings", "embeddings", "Embedding", "embedding"];

    /// <summary>The token layout plus loaded embeddings for one prompt; owns the loaded tensors, so dispose after encoding.</summary>
    public sealed class Plan : IDisposable
    {
        /// <summary>CLIP token ids <c>[MaxLength]</c> — SOT, interleaved text/placeholder ids, EOT, EOT-pad.</summary>
        public int[] TokenIds { get; set; } = [];

        /// <summary>Position of the EOT token, for pooled-output extraction.</summary>
        public int EosPosition { get; set; }

        /// <summary>Per embed occurrence: its first placeholder id plus the loaded <c>[N, hidden]</c> tensor keyed by hidden size.</summary>
        internal List<(int StartId, Dictionary<int, Tensor> ByHidden)> Occurrences { get; } = [];

        private readonly List<Tensor> _owned = [];

        internal void AddOwned(Tensor t) => _owned.Add(t);

        /// <summary>The <c>{placeholderId → [hidden] vector}</c> map for one encoder's hidden size; empty when no occurrence has a tensor at that size (e.g. an SD1.5-only embed under an SDXL CLIP-G request).</summary>
        public Dictionary<int, Tensor> InlineMap(int hiddenSize)
        {
            Dictionary<int, Tensor> merged = [];
            foreach ((int startId, Dictionary<int, Tensor> byHidden) in Occurrences)
            {
                if (byHidden.TryGetValue(hiddenSize, out Tensor? emb))
                {
                    (Dictionary<int, Tensor> map, _) = TextualInversion.BuildInlineMap(emb, startId);
                    foreach (KeyValuePair<int, Tensor> kv in map)
                    {
                        merged[kv.Key] = kv.Value;
                    }
                }
            }
            return merged;
        }

        /// <summary>Frees the loaded embedding tensors.</summary>
        public void Dispose()
        {
            foreach (Tensor t in _owned)
            {
                t.Dispose();
            }
            _owned.Clear();
        }
    }

    /// <summary>Builds the embed plan for <paramref name="prompt"/>, or null when it has no embed markers. <paramref name="hiddenSizes"/> are the encoder hidden sizes to load per embed ([768] for SD1.5, [768, 1280] for SDXL). Unresolvable or partially-loadable embeds are skipped. <paramref name="startPlaceholderId"/> defaults to just past the real vocab; callers resolving BOTH a positive and a negative prompt into the same batched <c>EncodePenultimate</c> call (see <see cref="BuildDualClipSchedule"/>) must give the negative plan a disjoint range — <c>inlineEmbeddings</c> is one dictionary shared across every row, so two different embeds landing on the same placeholder id would silently pick whichever one the dictionary happened to keep.</summary>
    public static Plan? Resolve(string? prompt, ClipTokenizer tokenizer, int[] hiddenSizes, int startPlaceholderId = ClipTokenizer.VocabSize)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(hiddenSizes);
        if (string.IsNullOrEmpty(prompt) || prompt.IndexOf('\u0000') < 0)
        {
            return null;
        }
        MatchCollection matches = _markerRx.Matches(prompt);
        if (matches.Count == 0)
        {
            return null;
        }

        Plan plan = new Plan();
        List<int> tokens = new List<int>(ClipTokenizer.MaxLength) { ClipTokenizer.StartOfTextId };
        int nextPlaceholder = startPlaceholderId; // defaults to 49408+, past the real vocab
        int cursor = 0;
        foreach (Match m in matches)
        {
            AppendRaw(tokens, tokenizer, prompt[cursor..m.Index]);
            cursor = m.Index + m.Length;

            string? path = ResolveEmbedPath(m.Groups[1].Value);
            if (path is null)
            {
                continue;
            }
            Dictionary<int, Tensor> byHidden = [];
            int n = 0;
            foreach (int h in hiddenSizes)
            {
                try
                {
                    Tensor emb = TextualInversion.Load(path, h);
                    plan.AddOwned(emb);
                    byHidden[h] = emb;
                    n = (int)emb.Shape[0];
                }
                catch (Exception ex)
                {
                    // This hidden size isn't present in the file (e.g. an SD1.5 embed has no CLIP-G tensor) — skip it.
                    Logs.Verbose($"[Features][Embed] '{path}' has no {h}-d tensor ({ex.GetType().Name}); skipping that encoder.");
                }
            }
            // SAFETY: only inject if the embed loaded for EVERY requested hidden size. Otherwise a placeholder id would be
            // missing from one encoder's inline map and EmbedTokens would fall through to the normal token-embedding lookup
            // at id >= VocabSize → out-of-bounds read (native crash). Skip partial embeds.
            if (n <= 0 || byHidden.Count != hiddenSizes.Length)
            {
                continue;
            }
            for (int r = 0; r < n; r++)
            {
                tokens.Add(nextPlaceholder + r);
            }
            plan.Occurrences.Add((nextPlaceholder, byHidden));
            nextPlaceholder += n;
        }
        AppendRaw(tokens, tokenizer, prompt[cursor..]);

        // Reserve the final slot for EOT, then pad with EOT (CLIP pads with EOT, not zero).
        int limit = ClipTokenizer.MaxLength - 1;
        if (tokens.Count > limit)
        {
            tokens.RemoveRange(limit, tokens.Count - limit);
        }
        plan.EosPosition = tokens.Count;
        tokens.Add(ClipTokenizer.EndOfTextId);
        while (tokens.Count < ClipTokenizer.MaxLength)
        {
            tokens.Add(ClipTokenizer.EndOfTextId);
        }
        plan.TokenIds = [.. tokens];

        if (plan.Occurrences.Count == 0)
        {
            plan.Dispose();
            return null;
        }
        return plan;
    }

    /// <summary>Removes the <c>\0swarmembed:…\0end</c> markers from a prompt, for the plain token path that feeds the pooled vector.</summary>
    public static string? StripMarkers(string? prompt)
        => string.IsNullOrEmpty(prompt) ? prompt : _markerRx.Replace(prompt, "");

    /// <summary>Builds an SDXL dual-CLIP <c>[2, S, 2048]</c> conditioning schedule (uncond, cond) from an embed plan: penultimate CLIP-L (768) + CLIP-G (1280), each encoded with its inline-embedding map so the learned vectors occupy the placeholder positions. Matches the SDXL pipeline's plain textEmbeddings shape, so it is passed as a conditioning-schedule override while the pooled vector stays on the pipeline's plain path. <para><paramref name="negativePlan"/> resolves embed markers in the negative prompt too — the common real-world shape for a "bad-hands"/"easynegative"-style embed. It must have been <see cref="Resolve"/>d with a <c>startPlaceholderId</c> disjoint from <paramref name="plan"/>'s (both maps below get merged into one dictionary shared across the whole batch). Either plan may be null — a null one falls back to that row's plain-encoded prompt, so a caller with an embed on only one side still gets the other side resolved.</para></summary>
    public static ConditioningSchedule BuildDualClipSchedule(
        IBackend backend, ClipTextEncoder clipL, ClipTextEncoder clipG, ClipTokenizer tokenizer,
        Plan? plan, string? positive, string? negative, int layersFromEnd, Plan? negativePlan = null)
    {
        ArgumentNullException.ThrowIfNull(clipL);
        ArgumentNullException.ThrowIfNull(clipG);
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (plan is null && negativePlan is null)
        {
            throw new ArgumentException($"{nameof(BuildDualClipSchedule)} requires at least one of {nameof(plan)}/{nameof(negativePlan)} to be non-null.");
        }

        int[] negTokens = negativePlan?.TokenIds ?? tokenizer.Encode(negative ?? "");
        int negEos = negativePlan?.EosPosition ?? ClipTokenizer.FindEosPosition(negTokens);
        int[] posTokens = plan?.TokenIds ?? tokenizer.Encode(positive ?? "");
        int posEos = plan?.EosPosition ?? ClipTokenizer.FindEosPosition(posTokens);
        int[][] batch = [negTokens, posTokens];
        int[] eos = [negEos, posEos];

        Dictionary<int, Tensor> mapL = plan?.InlineMap(768) ?? [];
        Dictionary<int, Tensor> mapG = plan?.InlineMap(1280) ?? [];
        if (negativePlan is not null)
        {
            foreach (KeyValuePair<int, Tensor> kv in negativePlan.InlineMap(768)) mapL[kv.Key] = kv.Value;
            foreach (KeyValuePair<int, Tensor> kv in negativePlan.InlineMap(1280)) mapG[kv.Key] = kv.Value;
        }

        (Tensor lHidden, Tensor? lPooled) = clipL.EncodePenultimate(backend, batch, eos, layersFromEnd, mapL);
        lPooled?.Dispose();
        (Tensor gHidden, Tensor? gPooled) = clipG.EncodePenultimate(backend, batch, eos, layersFromEnd, mapG);
        gPooled?.Dispose();

        Tensor concat = CfgHelper.ConcatLastDim(lHidden, gHidden);
        lHidden.Dispose();
        gHidden.Dispose();
        return new ConditioningSchedule { Variants = [concat], IndexForStep = static (_, _) => 0 };
    }

    private static void AppendRaw(List<int> tokens, ClipTokenizer tokenizer, string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return;
        }
        foreach (int id in tokenizer.EncodeRaw(text))
        {
            tokens.Add(id);
        }
    }

    /// <summary>Engine-native embedding lookup: resolves the embed name to a file under the models root's embeddings folders.</summary>
    private static string? ResolveEmbedPath(string name) => ModelFileLocator.Find(name, _folders);
}
