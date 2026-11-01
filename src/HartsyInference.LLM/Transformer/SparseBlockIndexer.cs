using System.Runtime.CompilerServices;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>MiniMax sparse attention's block selection: every query scores all earlier index keys, max-pools the scores per block of keys, and keeps its top blocks plus the blocks right before it.</summary>
public static class SparseBlockIndexer
{
    private static readonly ConditionalWeakTable<IKvCache, IndexKeyStore> Stores = new();

    /// <summary>Appends this step's index keys <c>[tNew * keyLength]</c> to <paramref name="cache"/>'s sequence and returns every key so far; <paramref name="posStart"/> 0 starts the sequence over.</summary>
    /// <exception cref="InvalidOperationException">The cache was rolled back or re-used, so its index keys no longer line up with the sequence.</exception>
    public static ReadOnlySpan<float> AppendKeys(IKvCache cache, int layer, int posStart, ReadOnlySpan<float> stepKeys, int keyLength)
    {
        IndexKeyStore store = Stores.GetValue(cache, static _ => new IndexKeyStore());
        return store.Append(layer, posStart, stepKeys, keyLength);
    }

    /// <summary>Builds the additive attention bias <c>[1, heads, t, keyCount]</c>: 0 for keys in a selected block of the query's indexer group, −∞ elsewhere. Causality stays with the attention mask.</summary>
    /// <param name="queries">Normed and rotated index queries <c>[t, indexHeads, headDim]</c>.</param>
    /// <param name="keys">Every index key so far <c>[keyCount, headDim]</c>.</param>
    public static Tensor BuildBias(ReadOnlySpan<float> queries, ReadOnlySpan<float> keys, int t, int posStart, int numHeads, IndexerConfig cfg)
    {
        int headDim = cfg.KeyLength;
        int keyCount = keys.Length / headDim;
        int blocks = (keyCount + cfg.BlockSize - 1) / cfg.BlockSize;
        int groupSize = numHeads / cfg.Heads;
        int topK = Math.Min(cfg.TopKBlocks, blocks);
        Tensor bias = new(new TensorShape(1, numHeads, t, keyCount), DType.F32);
        Span<float> output = bias.AsSpan<float>();
        output.Fill(float.NegativeInfinity);
        float[] blockScores = new float[blocks];
        int[] order = new int[blocks];
        for (int h = 0; h < cfg.Heads; h++)
        {
            for (int i = 0; i < t; i++)
            {
                int position = posStart + i;
                ReadOnlySpan<float> q = queries.Slice((i * cfg.Heads + h) * headDim, headDim);
                Array.Fill(blockScores, float.NegativeInfinity);
                int visible = Math.Min(keyCount, position + 1);
                for (int j = 0; j < visible; j++)
                {
                    float score = 0f;
                    ReadOnlySpan<float> k = keys.Slice(j * headDim, headDim);
                    for (int d = 0; d < headDim; d++) score += q[d] * k[d];
                    int block = j / cfg.BlockSize;
                    if (score > blockScores[block]) blockScores[block] = score;
                }
                int queryBlock = position / cfg.BlockSize;
                for (int l = 0; l < cfg.LocalBlocks; l++) blockScores[Math.Max(queryBlock - l, 0)] = float.PositiveInfinity;
                for (int b = 0; b < blocks; b++) order[b] = b;
                float[] scores = blockScores;
                Array.Sort(order, (x, y) => scores[y] != scores[x] ? scores[y].CompareTo(scores[x]) : x.CompareTo(y));
                for (int s = 0; s < topK; s++)
                {
                    int block = order[s];
                    if (float.IsNegativeInfinity(blockScores[block])) break;
                    int begin = block * cfg.BlockSize;
                    int end = Math.Min(begin + cfg.BlockSize, keyCount);
                    for (int g = 0; g < groupSize; g++)
                    {
                        int row = ((h * groupSize + g) * t + i) * keyCount;
                        output.Slice(row + begin, end - begin).Clear();
                    }
                }
            }
        }
        return bias;
    }

    private sealed class IndexKeyStore
    {
        private readonly Dictionary<int, (float[] Data, int Length)> _layers = [];

        public ReadOnlySpan<float> Append(int layer, int posStart, ReadOnlySpan<float> stepKeys, int keyLength)
        {
            (float[] data, int length) = posStart == 0 || !_layers.TryGetValue(layer, out (float[] Data, int Length) held) ? ([], 0) : held;
            if (length != posStart * keyLength)
            {
                throw new InvalidOperationException($"The sparse-attention index keys of layer {layer} hold {length / keyLength} tokens but the step starts at {posStart}; rolling the cache back is not supported.");
            }
            if (data.Length < length + stepKeys.Length) Array.Resize(ref data, Math.Max(data.Length * 2, length + stepKeys.Length));
            stepKeys.CopyTo(data.AsSpan(length));
            length += stepKeys.Length;
            _layers[layer] = (data, length);
            return data.AsSpan(0, length);
        }
    }
}
