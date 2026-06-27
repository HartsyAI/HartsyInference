using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.LLM.Embeddings;

namespace HartsyInference.TextGen.Cli;

/// <summary>Text-embedding harness for BERT-family GGUFs (bge / all-MiniLM / nomic). Encodes a sentence (or exact
/// token ids via HARTSY_EMBED_IDS, for reference parity) to a pooled, L2-normalized embedding.</summary>
public static class EmbedRunner
{
    public static int Run(string backendName, string text)
    {
        string? path = Environment.GetEnvironmentVariable("HARTSY_MODEL_DIR");
        if (path is null || !File.Exists(path)) { Console.Error.WriteLine("Set HARTSY_MODEL_DIR to a BERT embedding .gguf."); return 1; }

        using IBackend backend = backendName == "cpu"
            ? new CpuBackend()
            : new CudaBackend(deviceOrdinal: 0, ptxDir: Path.Combine(AppContext.BaseDirectory, "Ptx"));

        using BertEmbeddingModel model = BertEmbeddingModel.Load(path);
        Console.WriteLine($"=== embed ({backendName}) {Path.GetFileName(path)} ===");
        Console.WriteLine($"  hidden={model.Hidden} layers={model.NumLayers} heads={model.NumHeads} pooling={model.PoolingType}");

        // HARTSY_EMBED_IDS: exact token ids (reference parity). HARTSY_EMBED_SEMANTIC: run a cosine-similarity demo
        // over a few sentences (needs the GGUF vocab). Otherwise tokenize the single `text` and print its embedding.
        string? idsEnv = Environment.GetEnvironmentVariable("HARTSY_EMBED_IDS");
        if (idsEnv is not null)
        {
            int[] ids = idsEnv.Split(',').Select(s => int.Parse(s.Trim())).ToArray();
            Console.WriteLine($"  ids[{ids.Length}] = [{string.Join(",", ids)}]");
            float[] emb = model.Encode(backend, ids);
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

        if (model.Vocab is null) { Console.Error.WriteLine("GGUF has no tokenizer.ggml.tokens; pass HARTSY_EMBED_IDS."); return 1; }
        // Build the WordPiece tokenizer from the GGUF vocab (bge/MiniLM are uncased).
        using MemoryStream vocabStream = new(System.Text.Encoding.UTF8.GetBytes(string.Join('\n', model.Vocab)));
        using HartsyInference.Tokenizers.BertWordPieceTokenizer tokenizer = new(vocabStream, lowerCase: true);
        int cls = Array.IndexOf(model.Vocab, "[CLS]"), sep = Array.IndexOf(model.Vocab, "[SEP]");
        float[] Embed(string t)
        {
            int[] ids = [cls, .. tokenizer.EncodeRaw(t), sep];
            return model.Encode(backend, ids);
        }

        if (Environment.GetEnvironmentVariable("HARTSY_EMBED_SEMANTIC") == "1")
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
