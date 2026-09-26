using System.Text;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Converts a tiktoken rank file (<c>&lt;base64 bytes&gt; &lt;rank&gt;</c> per line) into a HuggingFace
/// <c>tokenizers</c> JSON: a byte-level BPE whose vocab is the ranks and whose merges are recovered from them.
/// <para>Byte-for-byte the output of <c>transformers</c>' <c>TikTokenConverter</c> serialized by <c>tokenizers</c>
/// (compact, vocab in id order, merges as pairs), which is what the Comfy-Org repacks embed — so a bundle built from
/// the upstream <c>.tiktoken</c> carries the identical tokenizer.</para></summary>
public static class TiktokenConverter
{
    /// <summary>Builds the tokenizer JSON as UTF-8 bytes.</summary>
    /// <param name="splitPattern">The pre-tokenizer regex (tiktoken's <c>pat_str</c>).</param>
    /// <param name="normalizer">A <c>tokenizers</c> normalizer type such as <c>NFC</c>, or null for none.</param>
    public static byte[] ToHuggingFaceJson(string tiktokenPath, string splitPattern, string? normalizer = "NFC")
    {
        List<byte[]> byRank = [];
        Dictionary<string, int> rankOf = new(StringComparer.Ordinal);
        foreach (string line in File.ReadLines(tiktokenPath))
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;
            int space = line.IndexOf(' ');
            byte[] token = Convert.FromBase64String(line[..space]);
            int rank = int.Parse(line[(space + 1)..], System.Globalization.CultureInfo.InvariantCulture);
            if (rank != byRank.Count)
                throw new InvalidDataException($"'{tiktokenPath}' ranks are not dense and ordered (expected {byRank.Count}, got {rank}).");
            byRank.Add(token);
            rankOf[Latin1(token)] = rank;
        }

        // TikTokenConverter: every split of a token into two ranked pieces is a merge; within a token the candidates
        // order by (rank(left), rank(right)); overall they order, stably, by the merged token's rank.
        List<(byte[] Left, byte[] Right)> merges = [];
        for (int rank = 0; rank < byRank.Count; rank++)
        {
            byte[] token = byRank[rank];
            if (token.Length == 1)
                continue;
            List<(byte[] L, byte[] R, int Rl, int Rr)> local = [];
            for (int i = 1; i < token.Length; i++)
            {
                byte[] l = token[..i], r = token[i..];
                if (rankOf.TryGetValue(Latin1(l), out int rl) && rankOf.TryGetValue(Latin1(r), out int rr))
                    local.Add((l, r, rl, rr));
            }
            foreach ((byte[] l, byte[] r, int _, int _) in local.OrderBy(x => x.Rl).ThenBy(x => x.Rr))
                merges.Add((l, r));
        }

        StringBuilder json = new(capacity: 8 << 20);
        json.Append("{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\"normalizer\":");
        json.Append(normalizer is null ? "null" : "{\"type\":" + Quote(normalizer) + "}");
        json.Append(",\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Split\",\"pattern\":{\"Regex\":");
        json.Append(Quote(splitPattern));
        json.Append("},\"behavior\":\"Isolated\",\"invert\":false},{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":false}]}");
        json.Append(",\"post_processor\":null,\"decoder\":{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":true,\"use_regex\":true}");
        json.Append(",\"model\":{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,\"continuing_subword_prefix\":null,\"end_of_word_suffix\":null,");
        json.Append("\"fuse_unk\":false,\"byte_fallback\":false,\"ignore_merges\":true,\"vocab\":{");
        for (int rank = 0; rank < byRank.Count; rank++)
        {
            if (rank > 0)
                json.Append(',');
            json.Append(Quote(ByteLevelCodec.EncodeBytes(byRank[rank]))).Append(':').Append(rank);
        }
        json.Append("},\"merges\":[");
        for (int i = 0; i < merges.Count; i++)
        {
            if (i > 0)
                json.Append(',');
            json.Append('[').Append(Quote(ByteLevelCodec.EncodeBytes(merges[i].Left))).Append(',')
                .Append(Quote(ByteLevelCodec.EncodeBytes(merges[i].Right))).Append(']');
        }
        json.Append("]}}");
        return Encoding.UTF8.GetBytes(json.ToString());
    }

    private static string Latin1(byte[] bytes) => Encoding.Latin1.GetString(bytes);

    // serde_json escaping: quote, backslash and control characters only; everything else stays raw UTF-8.
    private static string Quote(string value)
    {
        StringBuilder sb = new(value.Length + 2);
        sb.Append('"');
        foreach (char c in value)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                case '\b': sb.Append("\\b"); break;
                case '\f': sb.Append("\\f"); break;
                default:
                    if (c < 0x20)
                        sb.Append("\\u").Append(((int)c).ToString("x4", System.Globalization.CultureInfo.InvariantCulture));
                    else
                        sb.Append(c);
                    break;
            }
        }
        return sb.Append('"').ToString();
    }
}
