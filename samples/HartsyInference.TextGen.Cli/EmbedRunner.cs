using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.LLM.Embeddings;
using HartsyInference.ModelHandler.Gguf;

namespace HartsyInference.TextGen.Cli;

/// <summary>Text-embedding harness. Routes by GGUF architecture: BERT-family (bge / all-MiniLM / nomic) →
/// <see cref="BertEmbeddingModel"/>; decoder-family (Qwen3-Embedding / gte-Qwen2 / e5-mistral) →
/// <see cref="DecoderEmbeddingModel"/>. HARTSY_EMBED_IDS feeds exact token ids for reference parity;
/// HARTSY_EMBED_SEMANTIC runs a cosine-similarity demo.</summary>
public static class EmbedRunner
{
    public static int Run(string backendName, string text)
    {
        string? path = Environment.GetEnvironmentVariable("HARTSY_MODEL_DIR");
        if (path is null || !File.Exists(path)) { Console.Error.WriteLine("Set HARTSY_MODEL_DIR to an embedding .gguf."); return 1; }

        // Peek the architecture to pick the encoder.
        string arch;
        using (GgufLoader probe = new()) { probe.Load(path); arch = (probe.Metadata.GetString("general.architecture") ?? "").ToLowerInvariant(); }
        bool isBert = arch.Contains("bert");

        using IBackend backend = backendName == "cpu"
            ? new CpuBackend()
            : new CudaBackend(deviceOrdinal: 0, ptxDir: Path.Combine(AppContext.BaseDirectory, "Ptx"));

        string? idsEnv = Environment.GetEnvironmentVariable("HARTSY_EMBED_IDS");
        bool semantic = Environment.GetEnvironmentVariable("HARTSY_EMBED_SEMANTIC") == "1";

        // Encoder + tokenizer plumbing, abstracted over the two model types.
        Func<IReadOnlyList<int>, float[]> encode;
        Func<string, IReadOnlyList<int>>? tokenize;
        int hidden, pooling;
        IDisposable model;
        if (isBert)
        {
            BertEmbeddingModel m = BertEmbeddingModel.Load(path); model = m;
            hidden = m.Hidden; pooling = m.PoolingType;
            encode = ids => m.Encode(backend, ids);
            tokenize = m.Vocab is null ? null : Tok;
            IReadOnlyList<int> Tok(string t)
            {
                using MemoryStream vs = new(System.Text.Encoding.UTF8.GetBytes(string.Join('\n', m.Vocab!)));
                using HartsyInference.Tokenizers.BertWordPieceTokenizer tk = new(vs, lowerCase: true);
                int cls = Array.IndexOf(m.Vocab!, "[CLS]"), sep = Array.IndexOf(m.Vocab!, "[SEP]");
                return [cls, .. tk.EncodeRaw(t), sep];
            }
        }
        else
        {
            DecoderEmbeddingModel m = DecoderEmbeddingModel.Load(path); model = m;
            hidden = m.Hidden; pooling = m.PoolingType;
            encode = ids => m.Encode(backend, ids);
            tokenize = t => m.Tokenizer.Encode(t, addSpecial: true);
        }

        using (model)
        {
            Console.WriteLine($"=== embed ({backendName}) {Path.GetFileName(path)} ===");
            Console.WriteLine($"  arch={arch} type={(isBert ? "bert" : "decoder")} hidden={hidden} pooling={pooling}");

            // Reranker (cross-encoder) score path: HARTSY_RERANK=1 with HARTSY_EMBED_IDS (a [query]+[doc] pair).
            if (Environment.GetEnvironmentVariable("HARTSY_RERANK") == "1" && model is BertEmbeddingModel rm && rm.HasRerankHead && idsEnv is not null)
            {
                int[] rids = idsEnv.Split(',').Select(s => int.Parse(s.Trim())).ToArray();
                float logit = rm.Score(backend, rids);
                Console.WriteLine($"  ids[{rids.Length}] → rerank logit = {logit:F6}  (sigmoid={1f / (1f + MathF.Exp(-logit)):F4})");
                return 0;
            }

            if (idsEnv is not null)
            {
                int[] ids = idsEnv.Split(',').Select(s => int.Parse(s.Trim())).ToArray();
                Console.WriteLine($"  ids[{ids.Length}] = [{string.Join(",", ids)}]");
                float[] emb = encode(ids);
                Console.WriteLine($"  emb[{emb.Length}] first8 = [{string.Join(", ", emb.Take(8).Select(f => f.ToString("F4")))}]");
                string? dump = Environment.GetEnvironmentVariable("HARTSY_EMBED_DUMP");
                if (dump is not null)
                {
                    byte[] bytes = new byte[emb.Length * 4];
                    Buffer.BlockCopy(emb, 0, bytes, 0, bytes.Length);
                    File.WriteAllBytes(dump, bytes);
                    Console.WriteLine($"  dumped → {dump}");
                }
                return 0;
            }

            if (tokenize is null) { Console.Error.WriteLine("No tokenizer available; pass HARTSY_EMBED_IDS."); return 1; }
            float[] Embed(string t) => encode(tokenize(t));

            if (semantic)
            {
                string[] sents = ["a photo of a cat", "a picture of a kitten", "a red sports car"];
                float[][] v = sents.Select(Embed).ToArray();
                float Cos(float[] a, float[] b) { float d = 0; for (int i = 0; i < a.Length; i++) d += a[i] * b[i]; return d; }
                Console.WriteLine("  === semantic cosine similarity ===");
                Console.WriteLine($"  cos(cat, kitten) = {Cos(v[0], v[1]):F4}   (should be HIGH)");
                Console.WriteLine($"  cos(cat, car)    = {Cos(v[0], v[2]):F4}   (should be LOW)");
                return 0;
            }

            float[] e = Embed(text);
            Console.WriteLine($"  text=\"{text}\" → emb[{e.Length}] first8 = [{string.Join(", ", e.Take(8).Select(f => f.ToString("F4")))}]");
            return 0;
        }
    }
}
