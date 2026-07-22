using System.Collections.Generic;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>Turns parsed <see cref="WeightedSpan"/> chunks into padded CLIP token-id chunks plus a parallel per-token weight chunk, ready for <c>ClipTextEncoder.EncodeWeighted</c>. Mirrors ComfyUI/A1111 chunking: each <c>&lt;break&gt;</c> chunk and every 75-token overflow becomes its own SOT..EOT-wrapped 77-token chunk.</summary>
public static class WeightedPromptTokenizer
{
    /// <summary>Parses emphasis + <c>&lt;break&gt;</c> from a raw prompt, then tokenizes into id/weight chunks.</summary>
    public static (IReadOnlyList<int[]> Ids, IReadOnlyList<float[]> Weights) Tokenize(ClipTokenizer tokenizer, string prompt)
    {
        return Tokenize(tokenizer, PromptWeighting.ParseChunks(prompt));
    }

    /// <summary>Tokenizes pre-parsed break-chunks into id/weight chunks. Each break-chunk yields at least one 77-token chunk (empty chunks become a bare SOT..EOT).</summary>
    public static (IReadOnlyList<int[]> Ids, IReadOnlyList<float[]> Weights) Tokenize(
        ClipTokenizer tokenizer, IReadOnlyList<IReadOnlyList<WeightedSpan>> breakChunks)
    {
        if (tokenizer is null)
        {
            throw new ArgumentNullException(nameof(tokenizer));
        }
        const int contentLength = ClipTokenizer.MaxLength - 2;
        List<int[]> idChunks = new List<int[]>();
        List<float[]> weightChunks = new List<float[]>();
        foreach (IReadOnlyList<WeightedSpan> breakChunk in breakChunks)
        {
            List<int> ids = new List<int>();
            List<float> weights = new List<float>();
            foreach (WeightedSpan span in breakChunk)
            {
                IReadOnlyList<int> spanIds = tokenizer.EncodeRaw(span.Text);
                for (int i = 0; i < spanIds.Count; i++)
                {
                    ids.Add(spanIds[i]);
                    weights.Add(span.Weight);
                }
            }
            int pos = 0;
            do
            {
                int take = Math.Min(contentLength, ids.Count - pos);
                int[] idChunk = new int[ClipTokenizer.MaxLength];
                float[] weightChunk = new float[ClipTokenizer.MaxLength];
                idChunk[0] = ClipTokenizer.StartOfTextId;
                weightChunk[0] = 1f;
                for (int i = 0; i < take; i++)
                {
                    idChunk[i + 1] = ids[pos + i];
                    weightChunk[i + 1] = weights[pos + i];
                }
                for (int i = take + 1; i < ClipTokenizer.MaxLength; i++)
                {
                    idChunk[i] = ClipTokenizer.EndOfTextId;
                    weightChunk[i] = 1f;
                }
                idChunks.Add(idChunk);
                weightChunks.Add(weightChunk);
                pos += take;
            }
            while (pos < ids.Count);
        }
        if (idChunks.Count == 0)
        {
            int[] idChunk = new int[ClipTokenizer.MaxLength];
            float[] weightChunk = new float[ClipTokenizer.MaxLength];
            idChunk[0] = ClipTokenizer.StartOfTextId;
            weightChunk[0] = 1f;
            for (int i = 1; i < ClipTokenizer.MaxLength; i++)
            {
                idChunk[i] = ClipTokenizer.EndOfTextId;
                weightChunk[i] = 1f;
            }
            idChunks.Add(idChunk);
            weightChunks.Add(weightChunk);
        }
        return (idChunks, weightChunks);
    }
}
