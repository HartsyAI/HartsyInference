using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Ssm;

namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Drama 3's text-to-semantic model, ported from fish-speech's <c>generate_drama3</c>: the Qwen3.5-MoE hybrid backbone reads
/// the prompt rows (text embedding plus, on audio rows, the first codebooks' embeddings), the codebook tower is teacher-forced over the
/// prompt, and each decode step draws the next main token over eos plus the semantic band, lets the tower predict that frame's codes,
/// and feeds the frame back. The output head is cut to those candidate rows at load: the draw never reads any other logit.</summary>
public sealed unsafe class FishAudioDrama3Lm : IDisposable
{
    /// <summary>Default backbone prefill rows per forward; bounds the activations a long prompt holds at once.</summary>
    public const int DefaultPrefillChunk = 512;

    private const string BackbonePrefix = "model.language_model";
    private const string LmHeadKey = "lm_head.weight";

    private readonly FishAudioDrama3Config _config;
    private readonly Qwen35Model _backbone;
    private readonly FishAudioDrama3Tower _tower;
    private readonly Tensor _mainHead;      // lm_head rows of the candidates, [C, hidden]
    private readonly int[] _candidates;     // eos and the semantic band, ascending
    private int _disposed;

    /// <summary>Builds the model over <paramref name="weights"/> (checkpoint names: <c>model.language_model.*</c>, <c>lm_head.weight</c>,
    /// <c>audio_decoder*</c>), which must outlive it. <paramref name="capacity"/> is the most rows one sequence may reach.</summary>
    public FishAudioDrama3Lm(FishAudioDrama3Config config, IReadOnlyDictionary<string, Tensor> weights,
        int capacity = FishAudioDrama3Config.MaxPromptRows + FishAudioDrama3Config.DefaultMaxNewFrames, int prefillChunk = DefaultPrefillChunk)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(prefillChunk);
        Capacity = Math.Min(capacity, config.Backbone.MaxPositionEmbeddings);
        PrefillChunk = prefillChunk;
        _candidates = Candidates(config);
        _mainHead = CandidateRows(weights.TryGetValue(LmHeadKey, out Tensor? head) ? head
            : throw new KeyNotFoundException($"The Drama 3 checkpoint has no '{LmHeadKey}'."), _candidates, config.Backbone.VocabSize,
            config.Backbone.HiddenSize);
        try
        {
            _backbone = Qwen35Model.FromHuggingFace(weights, config.Backbone, Capacity, BackbonePrefix, lmHeadKey: null);
            _tower = new FishAudioDrama3Tower(config);
            _tower.LoadWeights(weights);
        }
        catch
        {
            _backbone?.Dispose();
            _tower?.Dispose();
            _mainHead.Dispose();
            throw;
        }
    }

    /// <summary>Most rows (prompt plus generated frames) one sequence may reach.</summary>
    public int Capacity { get; }

    /// <summary>Backbone prefill rows per forward.</summary>
    public int PrefillChunk { get; }

    /// <summary>Device bytes the KV caches grow by per sequence row, counted at F32 so the figure never undershoots: the backbone's
    /// full-attention layers (its Gated DeltaNet layers carry fixed-size state instead) and every tower layer.</summary>
    public long KvBytesPerRow
    {
        get
        {
            Qwen35HfConfig backbone = _config.Backbone;
            long fullAttention = 0;
            for (int i = 0; i < backbone.NumLayers; i++)
            {
                if (!backbone.IsLinearLayer(i)) fullAttention++;
            }
            FishAudioDrama3TowerConfig tower = _config.Tower;
            long towerLayers = tower.SharedLayers + tower.ResidualLayers;
            return (fullAttention * backbone.NumKvHeads * backbone.HeadDim + towerLayers * tower.NumKvHeads * tower.HeadDim) * 2 * sizeof(float);
        }
    }

    public FishAudioDrama3Config Config => _config;

    /// <summary>The backbone's weights grouped for <see cref="IBackend.PreloadWeightGroups"/> (each MoE layer's experts per projection,
    /// so the device-routed expert kernels find them resident), then the tower's and the head's.</summary>
    /// <param name="includeExperts">Include the routed experts; false when an expert offload owns them.</param>
    public IEnumerable<IReadOnlyList<Tensor>> EnumerateWeightGroups(bool includeExperts = true)
    {
        foreach (IReadOnlyList<Tensor> group in _backbone.EnumerateWeightGroups(includeExperts)) yield return group;
        foreach (Tensor t in _tower.EnumerateWeights()) yield return [t];
        yield return [_mainHead];
    }

    /// <summary>The backbone, for expert offload and residency planning.</summary>
    public Qwen35Model Backbone => _backbone;

    /// <summary>Generates frames after <paramref name="prompt"/> (<c>[rows, 1 + N]</c>) until eos or <paramref name="maxNewFrames"/>.
    /// Returns the codes <c>[N, frames]</c> without the eos frame; <paramref name="onFrame"/> sees each frame's codes as it is drawn.</summary>
    public FishAudioDrama3Generation Generate(IBackend backend, int[,] prompt, int maxNewFrames, FishAudioDrama3Sampler sampler,
        FishAudioDrama3FrameHandler? onFrame = null, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(sampler);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int rows = prompt.GetLength(0), width = 1 + _config.NumCodebooks, n = _config.NumCodebooks;
        if (prompt.GetLength(1) != width) throw new ArgumentException($"Prompt rows must hold {width} values.", nameof(prompt));
        if (rows == 0) throw new ArgumentException("The prompt is empty.", nameof(prompt));
        int budget = Math.Min(maxNewFrames, Capacity - rows);
        if (budget < 1) throw new ArgumentException($"The {rows}-row prompt leaves no room to generate within {Capacity} rows.", nameof(prompt));

        int[] frames = new int[(rows + budget) * width];
        Buffer.BlockCopy(prompt, 0, frames, 0, rows * width * sizeof(int));
        _backbone.ResetState();
        _tower.BeginSequence(rows + budget);
        Tensor last = Prefill(backend, frames, rows, null, cancel);
        try
        {
            int[] mains = new int[rows];
            for (int r = 0; r < rows; r++) mains[r] = frames[r * width];
            sampler.Prime(mains);
            float[] candidateLogits = new float[_candidates.Length];
            int[] codes = new int[n];
            int produced = 0;
            bool eos = false;
            for (int step = 0; step < budget; step++)
            {
                cancel.ThrowIfCancellationRequested();
                int position = rows - 1 + step;
                using (Tensor logits = new(new TensorShape(1, 1, _candidates.Length), DType.F32))
                {
                    backend.Linear(logits, last, _mainHead, null);
                    new ReadOnlySpan<float>((void*)logits.DataPointer, _candidates.Length).CopyTo(candidateLogits);
                }
                CheckFinite(candidateLogits);
                int main = sampler.SampleMain(candidateLogits, _candidates);
                if (main == _config.EosId)
                {
                    eos = true;
                    break;
                }
                _tower.DecodeFrame(backend, last, position, frames.AsSpan(position * width, width), main, codes, sampler);
                Span<int> next = frames.AsSpan((position + 1) * width, width);
                next[0] = main;
                codes.CopyTo(next[1..]);
                produced++;
                onFrame?.Invoke(codes);
                if (step + 1 == budget) break;
                Tensor nextHidden = Forward(backend, frames.AsSpan((position + 1) * width, width), 1);
                last.Dispose();
                last = nextHidden;
            }
            int[,] output = new int[n, produced];
            for (int f = 0; f < produced; f++)
            {
                for (int c = 0; c < n; c++) output[c, f] = frames[(rows + f) * width + 1 + c];
            }
            return new FishAudioDrama3Generation(output, eos);
        }
        finally
        {
            last.Dispose();
        }
    }

    /// <summary>Teacher-forced pass for verification and scoring: prefills <paramref name="prompt"/>, then for each of
    /// <paramref name="frames"/> (<c>[main, code 0, …, code N−1]</c>) records the candidate logits of its row, the tower's head logits
    /// with its codes forced, and the backbone hidden state after feeding it.</summary>
    public FishAudioDrama3Score Score(IBackend backend, int[,] prompt, IReadOnlyList<int[]> frames)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(frames);
        int rows = prompt.GetLength(0), width = 1 + _config.NumCodebooks, hiddenSize = _config.Backbone.HiddenSize;
        int heads = (_config.NumCodebooks - 1) * _config.CodebookSize;
        if (rows + frames.Count > Capacity) throw new ArgumentException("The prompt and frames exceed the model's capacity.", nameof(frames));
        int[] all = new int[(rows + frames.Count) * width];
        Buffer.BlockCopy(prompt, 0, all, 0, rows * width * sizeof(int));
        for (int f = 0; f < frames.Count; f++) frames[f].AsSpan(0, width).CopyTo(all.AsSpan((rows + f) * width));
        _backbone.ResetState();
        _tower.BeginSequence(rows + frames.Count);
        float[] promptHidden = new float[rows * hiddenSize];
        Tensor last = Prefill(backend, all, rows, (start, chunk) =>
            chunk.AsReadOnlySpan<float>().CopyTo(promptHidden.AsSpan(start * hiddenSize)), CancellationToken.None);
        try
        {
            float[][] candidateLogits = new float[frames.Count][], headLogits = new float[frames.Count][], hidden = new float[frames.Count][];
            for (int f = 0; f < frames.Count; f++)
            {
                int position = rows - 1 + f;
                candidateLogits[f] = new float[_candidates.Length];
                using (Tensor logits = new(new TensorShape(1, 1, _candidates.Length), DType.F32))
                {
                    backend.Linear(logits, last, _mainHead, null);
                    logits.AsReadOnlySpan<float>().CopyTo(candidateLogits[f]);
                }
                headLogits[f] = new float[heads];
                ReadOnlySpan<int> frame = all.AsSpan((position + 1) * width, width);
                _tower.ScoreFrame(backend, last, position, all.AsSpan(position * width, width), frame[0], frame[1..], headLogits[f]);
                Tensor next = Forward(backend, frame, 1);
                last.Dispose();
                last = next;
                hidden[f] = last.AsReadOnlySpan<float>().ToArray();
            }
            return new FishAudioDrama3Score(promptHidden, _candidates, candidateLogits, headLogits, hidden);
        }
        finally
        {
            last.Dispose();
        }
    }

    // Runs the prompt through the backbone in chunks, teacher-forcing the tower on every row but the last, and returns the last
    // row's hidden state [1, 1, hidden]; hiddenSink, when given, sees each chunk's hidden states with its first row.
    private Tensor Prefill(IBackend backend, int[] frames, int rows, Action<int, Tensor>? hiddenSink, CancellationToken cancel)
    {
        int width = 1 + _config.NumCodebooks;
        Tensor? last = null;
        for (int start = 0; start < rows; start += PrefillChunk)
        {
            cancel.ThrowIfCancellationRequested();
            int count = Math.Min(PrefillChunk, rows - start);
            using Tensor hidden = Forward(backend, frames.AsSpan(start * width, count * width), count);
            hiddenSink?.Invoke(start, hidden);
            int paired = Math.Min(start + count, rows - 1) - start;
            if (paired == count)
            {
                _tower.Prefill(backend, hidden, start, paired, frames.AsSpan(start * width), frames.AsSpan((start + 1) * width));
            }
            else
            {
                if (paired > 0)
                {
                    using Tensor head = Rows(backend, hidden, 0, paired);
                    _tower.Prefill(backend, head, start, paired, frames.AsSpan(start * width), frames.AsSpan((start + 1) * width));
                }
                last = Rows(backend, hidden, count - 1, 1);
            }
        }
        return last!;
    }

    // Backbone forward over count rows of frames, continuing the carried state; returns the final-norm hidden [1, count, hidden].
    private Tensor Forward(IBackend backend, ReadOnlySpan<int> frames, int count)
    {
        int width = 1 + _config.NumCodebooks;
        int[] ids = new int[count];
        for (int r = 0; r < count; r++) ids[r] = frames[r * width];
        using Tensor embeds = new(new TensorShape(1, count, _config.Backbone.HiddenSize), DType.F32);
        _backbone.EmbedLookup(embeds, ids);
        _tower.MixBackboneInput((float*)embeds.DataPointer, frames, count);
        return _backbone.ForwardEmbedsHidden(backend, embeds, count);
    }

    private static Tensor Rows(IBackend backend, Tensor hidden, int first, int count)
    {
        int[] index = new int[count];
        for (int i = 0; i < count; i++) index[i] = first + i;
        Tensor rows = new(new TensorShape(1, count, hidden.Shape[2]), DType.F32);
        backend.GatherRows(rows, hidden, index);
        return rows;
    }

    private static void CheckFinite(ReadOnlySpan<float> logits)
    {
        foreach (float v in logits)
        {
            if (!float.IsFinite(v)) throw new InvalidOperationException("Drama 3 produced non-finite logits; refusing to sample invalid audio.");
        }
    }

    /// <summary>Eos and every semantic id, ascending — the ids the main token is drawn over.</summary>
    private static int[] Candidates(FishAudioDrama3Config config)
    {
        int band = config.SemanticEndId - config.SemanticStartId + 1;
        int[] ids = new int[band + 1];
        for (int i = 0; i < band; i++) ids[i] = config.SemanticStartId + i;
        ids[band] = config.EosId;
        Array.Sort(ids);
        return ids;
    }

    // Copies the head rows of the candidates into an owned [C, hidden] tensor in the head's own dtype. The head must have the config's
    // shape, and every candidate a row in it, before any row is read through a pointer.
    private static Tensor CandidateRows(Tensor head, int[] candidates, int vocab, int hidden)
    {
        if (head.Shape.Rank != 2 || head.Shape[0] != vocab || head.Shape[1] != hidden)
        {
            throw new InvalidDataException($"The Drama 3 lm_head is {head.Shape}; the config expects [{vocab}, {hidden}].");
        }
        foreach (int id in candidates)
        {
            if ((uint)id >= (uint)vocab)
            {
                throw new InvalidDataException($"The Drama 3 config names token {id} (eos or the semantic band), outside its {vocab}-token vocabulary.");
            }
        }
        long rowBytes = head.DType.ComputeByteCount(hidden);
        Tensor rows = new(new TensorShape(candidates.Length, hidden), head.DType);
        byte* src = (byte*)head.DataPointer;
        byte* dst = (byte*)rows.DataPointer;
        for (int i = 0; i < candidates.Length; i++)
        {
            Buffer.MemoryCopy(src + candidates[i] * rowBytes, dst + i * rowBytes, rowBytes, rowBytes);
        }
        return rows;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _backbone.Dispose();
        _tower.Dispose();
        _mainHead.Dispose();
    }
}
