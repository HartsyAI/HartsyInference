using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Ssm;

namespace HartsyInference.Audio.Models.Haru;

/// <summary>Haru's language model: the Qwen3.5 text trunk plus the two audio projectors. One position is one 133 ms frame, and its
/// input is the sum of the model's text-token embedding and the projections of its own and the user's audio latents.</summary>
internal sealed unsafe class HaruLm : IDisposable
{
    private readonly HaruConfig _config;
    private readonly Qwen35Model _trunk;
    private readonly Tensor _embedTable;
    private readonly HaruAudioProjector _modelAudio;
    private readonly HaruAudioProjector _userAudio;
    private readonly float _latentBias;
    private readonly float _latentScale;
    private float[]? _streamRow;

    /// <summary>Builds the trunk and projectors over <paramref name="weights"/> (<c>lm.*</c>, <c>model_audio_proj.*</c>,
    /// <c>user_audio_proj.*</c>, <c>latent_scale</c>, <c>latent_bias</c>), which must outlive it. <paramref name="capacity"/> is the
    /// most positions a conversation may reach: prompt plus frames.</summary>
    internal HaruLm(HaruConfig config, IReadOnlyDictionary<string, Tensor> weights, int capacity)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        _config = config;
        Capacity = Math.Min(capacity, config.Lm.MaxPositionEmbeddings);
        _latentBias = Scalar(weights, "latent_bias");
        _latentScale = Scalar(weights, "latent_scale");
        _embedTable = weights.TryGetValue("lm.embed_tokens.weight", out Tensor? table)
            ? table : throw new KeyNotFoundException("The Haru checkpoint has no tensor 'lm.embed_tokens.weight'.");
        _trunk = Qwen35Model.FromHuggingFace(weights, config.Lm, Capacity, "lm", lmHeadKey: null);
        _modelAudio = new HaruAudioProjector(weights, "model_audio_proj", config.LatentDim, config.Lm.HiddenSize);
        _userAudio = new HaruAudioProjector(weights, "user_audio_proj", config.LatentDim, config.Lm.HiddenSize);
    }

    internal int HiddenSize => _config.Lm.HiddenSize;

    internal int VocabSize => _config.Lm.VocabSize;

    /// <summary>Most positions the conversation may reach: prompt plus frames.</summary>
    internal int Capacity { get; }

    /// <summary>Positions the trunk has consumed since the last <see cref="Reset"/>.</summary>
    internal int Position => _trunk.CurrentPosition;

    /// <summary>The trunk, for weight preloading and placement.</summary>
    internal Qwen35Model Trunk => _trunk;

    /// <summary>The raw codec latent of a normalized one, <c>x / scale - bias</c>.</summary>
    internal float Unscale(float normalized) => normalized / _latentScale - _latentBias;

    /// <summary>Clears the conversation: every layer's recurrent state and the attention cache.</summary>
    internal void Reset() => _trunk.ResetState();

    /// <summary>The input of <c>tokens.Length</c> positions as a fresh host <c>[1, T, H]</c> tensor the caller owns. A latent span holds
    /// <c>T * latentDim</c> raw (un-normalized) codec latents, or is empty when that stream has none; a mask marks the positions that
    /// carry the stream, or is empty when all do (an empty mask with latents means every position).</summary>
    internal Tensor Embed(IBackend backend, ReadOnlySpan<int> tokens, ReadOnlySpan<float> modelLatents, ReadOnlySpan<bool> modelMask,
        ReadOnlySpan<float> userLatents, ReadOnlySpan<bool> userMask)
    {
        int rows = tokens.Length;
        if (rows == 0)
        {
            throw new ArgumentException("A position needs a text token.", nameof(tokens));
        }
        Tensor embeds = new(new TensorShape(1, rows, HiddenSize), DType.F32);
        _trunk.EmbedLookup(embeds, tokens.ToArray());
        AddStream(backend, embeds, _modelAudio, modelLatents, modelMask, rows);
        AddStream(backend, embeds, _userAudio, userLatents, userMask, rows);
        return embeds;
    }

    /// <summary>Runs <paramref name="embeddings"/> (<c>[1, T, H]</c>, not consumed) on top of the conversation so far and returns the
    /// final-norm hidden state of every position, <c>[1, T, H]</c>, owned by the caller.</summary>
    internal Tensor Forward(IBackend backend, Tensor embeddings, int rows)
    {
        if (embeddings.Shape.Rank != 3 || embeddings.Shape[0] != 1 || embeddings.Shape[1] != rows || embeddings.Shape[2] != HiddenSize)
        {
            throw new ArgumentException($"embeddings must be [1, {rows}, {HiddenSize}], got {embeddings.Shape}.", nameof(embeddings));
        }
        return _trunk.ForwardEmbedsHidden(backend, embeddings, rows);
    }

    /// <summary>Text logits of the last <paramref name="rows"/> rows of <paramref name="hidden"/> (<c>[1, T, H]</c>, F32, <c>rows &lt;= T</c>)
    /// over the whole vocabulary, as a fresh <c>[1, rows, vocab]</c> F32 tensor. Only those rows are projected: after a long prompt chunk
    /// the frame loop needs the final position alone, and the full table is 248,320 floats a row.</summary>
    internal Tensor Logits(IBackend backend, Tensor hidden, int rows)
    {
        CheckHidden(hidden);
        long total = hidden.Shape[1];
        if (rows < 1 || rows > total)
        {
            throw new ArgumentOutOfRangeException(nameof(rows), rows, $"hidden has {total} rows.");
        }
        if (rows == total)
        {
            return _trunk.ProjectLogits(backend, hidden, rows);
        }
        int h = HiddenSize;
        using Tensor tail = new(new TensorShape(1, rows, h), DType.F32);
        new ReadOnlySpan<float>((float*)hidden.DataPointer + (total - rows) * h, rows * h).CopyTo(new Span<float>((float*)tail.DataPointer, rows * h));
        return _trunk.ProjectLogits(backend, tail, rows);
    }

    /// <summary>The logits of <paramref name="tokens"/> at row <paramref name="row"/> of <paramref name="hidden"/>, with the dot products
    /// taken in F32 or wider from the F32 embedding rows. The decision to stay silent or start speaking is finer than BF16 resolves at
    /// the magnitude of these logits.</summary>
    internal void StreamLogits(Tensor hidden, int row, ReadOnlySpan<int> tokens, Span<float> destination)
    {
        if (destination.Length < tokens.Length)
        {
            throw new ArgumentException("The destination is shorter than the token list.", nameof(destination));
        }
        CheckHidden(hidden);
        if ((uint)row >= (uint)hidden.Shape[1])
        {
            throw new ArgumentOutOfRangeException(nameof(row), row, $"hidden has {hidden.Shape[1]} rows.");
        }
        int h = HiddenSize;
        float* hiddenRow = (float*)hidden.DataPointer + (long)row * h;
        float[] embedding = _streamRow ??= new float[h];
        for (int i = 0; i < tokens.Length; i++)
        {
            if ((uint)tokens[i] >= (uint)VocabSize)
            {
                throw new ArgumentOutOfRangeException(nameof(tokens), tokens[i], $"The vocabulary has {VocabSize} tokens.");
            }
            fixed (float* e = embedding)
            {
                Qwen35Model.CopyEmbeddingRow(_embedTable, tokens[i], h, e);
            }
            double sum = 0;
            for (int c = 0; c < h; c++)
            {
                sum += (double)hiddenRow[c] * embedding[c];
            }
            destination[i] = (float)sum;
        }
    }

    internal IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _modelAudio.EnumerateWeights())
        {
            yield return t;
        }
        foreach (Tensor t in _userAudio.EnumerateWeights())
        {
            yield return t;
        }
    }

    public void Dispose()
    {
        _trunk.Dispose();
        _modelAudio.Dispose();
        _userAudio.Dispose();
    }

    private void CheckHidden(Tensor hidden)
    {
        if (hidden.Shape.Rank != 3 || hidden.Shape[0] != 1 || hidden.Shape[2] != HiddenSize || hidden.DType != DType.F32)
        {
            throw new ArgumentException($"hidden must be F32 [1, T, {HiddenSize}], got {hidden.DType} {hidden.Shape}.", nameof(hidden));
        }
    }

    private void AddStream(IBackend backend, Tensor embeds, HaruAudioProjector projector, ReadOnlySpan<float> latents,
        ReadOnlySpan<bool> mask, int rows)
    {
        if (latents.IsEmpty)
        {
            return;
        }
        int d = _config.LatentDim;
        if (latents.Length != rows * d || (!mask.IsEmpty && mask.Length != rows))
        {
            throw new ArgumentException($"A stream takes {rows} x {d} latents and a mask of {rows} (or none).", nameof(latents));
        }
        using Tensor normalized = new(new TensorShape(1, rows, d), DType.F32);
        float* n = (float*)normalized.DataPointer;
        for (int i = 0; i < latents.Length; i++)
        {
            n[i] = (latents[i] + _latentBias) * _latentScale;
        }
        using Tensor projected = projector.Forward(backend, normalized);
        float* p = (float*)projected.DataPointer;
        float* e = (float*)embeds.DataPointer;
        int h = HiddenSize;
        for (int r = 0; r < rows; r++)
        {
            if (!mask.IsEmpty && !mask[r])
            {
                continue;
            }
            for (int c = 0; c < h; c++)
            {
                e[(long)r * h + c] += p[(long)r * h + c];
            }
        }
    }

    private static float Scalar(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        if (!weights.TryGetValue(key, out Tensor? t))
        {
            throw new KeyNotFoundException($"The Haru checkpoint has no tensor '{key}'.");
        }
        if (t.ElementCount != 1)
        {
            throw new InvalidDataException($"'{key}' must hold a single value, it has {t.ElementCount}.");
        }
        Tensor f32 = TensorCasts.EnsureF32(t);
        float value = *(float*)f32.DataPointer;
        if (!ReferenceEquals(f32, t))
        {
            f32.Dispose();
        }
        return value;
    }
}
