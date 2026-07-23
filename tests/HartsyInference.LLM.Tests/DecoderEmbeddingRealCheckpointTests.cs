using System.Globalization;
using HartsyInference.Cpu;
using HartsyInference.LLM.Embeddings;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>Env-gated real-reference verification for <see cref="DecoderEmbeddingModel"/> — the class had zero
/// callers and no runnable correctness test anywhere in the repo before this pass (its old Python parity script
/// referenced a CLI hook that no longer exists). Skips cleanly unless <c>HARTSY_EMBED_GGUF_PATH</c> points at a
/// real Qwen3-Embedding GGUF. Compares against the full 1024-dim reference vector in
/// <c>Qwen3EmbeddingReference.txt</c>, captured from <c>transformers.AutoModel</c> (5.14.1) on CPU for the exact
/// same token ids (<c>tok('a photo of a cat.')</c> on Qwen/Qwen3-Embedding-0.6B →
/// <c>[64, 6548, 315, 264, 8251, 13, 151643]</c>, <c>F.normalize(out.last_hidden_state[0, -1], dim=0)</c>) —
/// feeding the SAME ids on both sides isolates model/pooling correctness from tokenizer correctness, same
/// discipline the removed <c>dump_decoder_embedding_ref.py</c> script used.</summary>
public sealed class DecoderEmbeddingRealCheckpointTests
{
    private readonly ITestOutputHelper _output;
    public DecoderEmbeddingRealCheckpointTests(ITestOutputHelper output) => _output = output;

    // tok('a photo of a cat.') on Qwen/Qwen3-Embedding-0.6B via transformers 5.14.1, captured 2026-07-22.
    private static readonly int[] ReferenceIds = [64, 6548, 315, 264, 8251, 13, 151643];

    [Fact]
    public void MatchesHfReference_ForFixedTokenIds()
    {
        string? path = Environment.GetEnvironmentVariable("HARTSY_EMBED_GGUF_PATH");
        if (string.IsNullOrEmpty(path))
        {
            _output.WriteLine("SKIPPED: HARTSY_EMBED_GGUF_PATH not set.");
            return;
        }

        string refPath = Path.Combine(AppContext.BaseDirectory, "Qwen3EmbeddingReference.txt");
        float[] reference = [.. File.ReadAllText(refPath).Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(s => float.Parse(s, CultureInfo.InvariantCulture))];
        Assert.Equal(1024, reference.Length);

        using DecoderEmbeddingModel model = DecoderEmbeddingModel.Load(path);
        using CpuBackend backend = new();
        float[] vector = model.Encode(backend, ReferenceIds);

        Assert.Equal(1024, model.Hidden);
        Assert.Equal(1024, vector.Length);

        for (int i = 0; i < 8; i++)
            _output.WriteLine($"[{i}] c# = {vector[i]:F4}  ref = {reference[i]:F4}");

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < vector.Length; i++)
        {
            dot += (double)vector[i] * reference[i];
            normA += (double)vector[i] * vector[i];
            normB += (double)reference[i] * reference[i];
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB) + 1e-12);
        _output.WriteLine($"Full 1024-dim cosine similarity vs. HF reference: {cosine:F6}");
        Assert.True(cosine >= 0.99, $"Cosine similarity {cosine:F6} is below the 0.99 bar.");
    }
}
