using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Ssm;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Drama 3's codebook-MTP audio decoder, ported from fish-speech's <c>FishOmniAudioDecoder</c>. Given the backbone's
/// hidden state for a row and the next frame's main token, it predicts that frame's codes 1..N−1: shared enrichment layers and then
/// one residual layer per codebook each mix a code embedding into the hidden state (<c>eh_proj([enorm(e); hnorm(h)])</c>) and run one
/// pre-norm transformer layer — gated attention (<c>wqkv</c> rows <c>[Q | gate | K | V]</c>), zero-centered norms, interleaved
/// partial RoPE — with its own KV cache over the sequence's rows. Prompt rows are teacher-forced with the frame that follows them.</summary>
public sealed unsafe class FishAudioDrama3Tower : IDisposable
{
    private const string CodebookEmbeddingsKey = "audio_decoder.codebook_embeddings.weight";
    private const string SharedPrefix = "audio_decoder_mtp_shared.layers";
    private const string ResidualPrefix = "audio_decoder_mtp.layers";
    private const string HeadPrefix = "audio_decoder_mtp.heads";

    private readonly FishAudioDrama3Config _config;
    private readonly FishAudioDrama3TowerConfig _tower;
    private readonly TowerLayer[] _shared;
    private readonly TowerLayer[] _residual;
    private Tensor _codebookEmbeddings = null!;   // [V·N, dim], host-gathered
    private Tensor[] _heads = [];                 // [V, dim] per codebook after the first
    private readonly float[] _row;                // one embedding row, gathered on the host
    private readonly int[] _code = new int[1];
    private readonly bool[] _semantic = new bool[1];
    private int _disposed;

    public FishAudioDrama3Tower(FishAudioDrama3Config config)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _tower = config.Tower;
        TransformerConfig block = new()
        {
            HiddenSize = _tower.Dim,
            NumLayers = 1,
            NumHeads = _tower.NumHeads,
            NumKvHeads = _tower.NumKvHeads,
            HeadDim = _tower.HeadDim,
            RotaryDim = _tower.RotaryDim,
            IntermediateSize = _tower.IntermediateSize,
            VocabSize = _tower.VocabSize,
            MaxPositionEmbeddings = config.Backbone.MaxPositionEmbeddings,
            RopeTheta = _tower.RopeBase,
            RmsNormEps = _tower.NormEps,
            AttentionBias = false,
            QkNorm = true,
            RmsNormAddOne = true,
            TieWordEmbeddings = false,
            Rope = RopeStyle.Interleaved,
            AttnOutputGate = true,
        };
        _row = new float[_tower.Dim];
        _shared = new TowerLayer[_tower.SharedLayers];
        _residual = new TowerLayer[_tower.ResidualLayers];
        for (int i = 0; i < _shared.Length; i++) _shared[i] = new TowerLayer(block, _tower);
        for (int i = 0; i < _residual.Length; i++) _residual[i] = new TowerLayer(block, _tower);
    }

    /// <summary>Loads the <c>audio_decoder*</c> tensors of a Drama 3 checkpoint; every tower parameter must be present.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        ArgumentNullException.ThrowIfNull(w);
        _codebookEmbeddings = Require(w, CodebookEmbeddingsKey, (long)_tower.VocabSize * _tower.NumCodebooks, _tower.Dim);
        for (int i = 0; i < _shared.Length; i++) _shared[i].LoadWeights(w, $"{SharedPrefix}.{i}", "semantic_embedding");
        for (int i = 0; i < _residual.Length; i++) _residual[i].LoadWeights(w, $"{ResidualPrefix}.{i}", "codebook_embedding");
        _heads = new Tensor[_residual.Length];
        for (int i = 0; i < _heads.Length; i++) _heads[i] = Require(w, $"{HeadPrefix}.{i}.weight", _tower.VocabSize, _tower.Dim);
    }

    /// <summary>The weights the backend reads; the code embedding tables are gathered on the host and stay off the device.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (TowerLayer layer in _shared) foreach (Tensor t in layer.EnumerateWeights()) yield return t;
        foreach (TowerLayer layer in _residual) foreach (Tensor t in layer.EnumerateWeights()) yield return t;
        foreach (Tensor head in _heads) yield return head;
    }

    /// <summary>Starts a sequence of at most <paramref name="capacity"/> rows: every layer gets an empty KV cache.</summary>
    public void BeginSequence(int capacity)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        foreach (TowerLayer layer in _shared) layer.BeginSequence(capacity);
        foreach (TowerLayer layer in _residual) layer.BeginSequence(capacity);
    }

    /// <summary>Turns text embeddings (<c>[count, dim]</c> host F32, in place) into the backbone input of the rows <paramref name="frames"/>
    /// (<c>[count, 1 + N]</c>): a semantic row adds the embeddings of codes 1..K and is scaled by <c>1/√(1 + K)</c>.</summary>
    public void MixBackboneInput(float* rows, ReadOnlySpan<int> frames, int count)
    {
        int dim = _tower.Dim, width = 1 + _tower.NumCodebooks, k = _tower.BackboneCodebooks;
        float scale = 1f / MathF.Sqrt(1 + k);
        for (int r = 0; r < count; r++)
        {
            ReadOnlySpan<int> frame = frames.Slice(r * width, width);
            if (!_config.IsSemantic(frame[0])) continue;
            float* row = rows + (long)r * dim;
            for (int c = 1; c <= k; c++) AddCodeEmbedding(row, frame, c);
            for (int j = 0; j < dim; j++) row[j] *= scale;
        }
    }

    /// <summary>Teacher-forces the caches of <paramref name="count"/> prompt rows from <paramref name="position"/>: row p is labelled with
    /// the frame after it, <paramref name="nextFrames"/>[p]. <paramref name="hidden"/> is the backbone's <c>[1, count, dim]</c> output.</summary>
    public void Prefill(IBackend backend, Tensor hidden, int position, int count, ReadOnlySpan<int> frames, ReadOnlySpan<int> nextFrames)
    {
        int width = 1 + _tower.NumCodebooks;
        int[] codes = new int[count];
        bool[] semantic = new bool[count];
        for (int r = 0; r < count; r++) semantic[r] = _config.IsSemantic(nextFrames[r * width]);
        for (int r = 0; r < count; r++) codes[r] = nextFrames[r * width + 1];
        Tensor h = RunShared(backend, hidden, codes, semantic, position, count, frames);
        try
        {
            for (int i = 0; i < _residual.Length; i++)
            {
                for (int r = 0; r < count; r++) codes[r] = nextFrames[r * width + 1 + i];
                Tensor next = _residual[i].Forward(backend, h, codes, semantic, position, count, null);
                h.Dispose();
                h = next;
            }
        }
        finally
        {
            h.Dispose();
        }
    }

    /// <summary>Predicts the codes of the frame after row <paramref name="position"/> whose main token is <paramref name="main"/>:
    /// writes codes 0..N−1 into <paramref name="codes"/>, sampling 1..N−1 from the heads. <paramref name="hidden"/> is the backbone's
    /// <c>[1, 1, dim]</c> output for the row and <paramref name="frame"/> the row itself.</summary>
    public void DecodeFrame(IBackend backend, Tensor hidden, int position, ReadOnlySpan<int> frame, int main, Span<int> codes,
        FishAudioDrama3Sampler sampler)
    {
        ArgumentNullException.ThrowIfNull(sampler);
        RunFrame(backend, hidden, position, frame, main, codes, sampler, default);
    }

    /// <summary>Teacher-forced <see cref="DecodeFrame"/>: feeds the frame's <paramref name="codes"/> (all N; code 0 follows from
    /// <paramref name="main"/> whatever is given) and writes each head's logits, <c>[N−1, V]</c>, into <paramref name="headLogits"/>.</summary>
    public void ScoreFrame(IBackend backend, Tensor hidden, int position, ReadOnlySpan<int> frame, int main, ReadOnlySpan<int> codes,
        Span<float> headLogits)
    {
        if (headLogits.Length != _residual.Length * _tower.VocabSize)
        {
            throw new ArgumentException($"Expected room for {_residual.Length} × {_tower.VocabSize} logits.", nameof(headLogits));
        }
        int[] forced = codes.ToArray();
        RunFrame(backend, hidden, position, frame, main, forced, null, headLogits);
    }

    private void RunFrame(IBackend backend, Tensor hidden, int position, ReadOnlySpan<int> frame, int main, Span<int> codes,
        FishAudioDrama3Sampler? sampler, Span<float> headLogits)
    {
        if (codes.Length != _tower.NumCodebooks) throw new ArgumentException($"Expected {_tower.NumCodebooks} codes.", nameof(codes));
        int[] code = _code;
        bool[] semantic = _semantic;
        code[0] = Math.Clamp(main - _config.SemanticStartId, 0, _tower.VocabSize - 1);
        semantic[0] = _config.IsSemantic(main);
        codes[0] = code[0];
        Tensor h = RunShared(backend, hidden, code, semantic, position, 1, frame);
        try
        {
            for (int i = 0; i < _residual.Length; i++)
            {
                Tensor next = _residual[i].Forward(backend, h, code, semantic, position, 1, null);
                h.Dispose();
                h = next;
                using Tensor logits = new(new TensorShape(1, 1, _tower.VocabSize), DType.F32);
                backend.Linear(logits, h, _heads[i], null);
                ReadOnlySpan<float> values = new((void*)logits.DataPointer, _tower.VocabSize);
                if (sampler is not null)
                {
                    codes[i + 1] = sampler.SampleCode(values);
                }
                else
                {
                    values.CopyTo(headLogits.Slice(i * _tower.VocabSize, _tower.VocabSize));
                }
                code[0] = codes[i + 1];
            }
        }
        finally
        {
            h.Dispose();
        }
    }

    // The shared layers; the first one also sees the embeddings of the codes the backbone input did not carry (codes K+1..N−1 of
    // the current rows, semantic rows only). Returns a new hidden state; the input is not consumed.
    private Tensor RunShared(IBackend backend, Tensor hidden, int[] codes, bool[] semantic, int position, int count, ReadOnlySpan<int> frames)
    {
        int dim = _tower.Dim, width = 1 + _tower.NumCodebooks;
        Tensor unseen = new(new TensorShape(1, count, dim), DType.F32);
        try
        {
            float* u = (float*)unseen.DataPointer;
            new Span<float>(u, count * dim).Clear();
            for (int r = 0; r < count; r++)
            {
                ReadOnlySpan<int> frame = frames.Slice(r * width, width);
                if (!_config.IsSemantic(frame[0])) continue;
                for (int c = 1 + _tower.BackboneCodebooks; c < _tower.NumCodebooks; c++) AddCodeEmbedding(u + (long)r * dim, frame, c);
            }
            Tensor h = _shared[0].Forward(backend, hidden, codes, semantic, position, count, unseen);
            for (int i = 1; i < _shared.Length; i++)
            {
                Tensor next = _shared[i].Forward(backend, h, codes, semantic, position, count, null);
                h.Dispose();
                h = next;
            }
            return h;
        }
        finally
        {
            unseen.Dispose();
        }
    }

    // row += codebook_embeddings[code_c + c·V] for the code of codebook c in frame (column 1 + c).
    private void AddCodeEmbedding(float* row, ReadOnlySpan<int> frame, int codebook)
    {
        int dim = _tower.Dim;
        int code = Math.Clamp(frame[1 + codebook], 0, _tower.VocabSize - 1);
        fixed (float* scratch = _row)
        {
            Qwen35Model.CopyEmbeddingRow(_codebookEmbeddings, code + codebook * _tower.VocabSize, dim, scratch);
            for (int j = 0; j < dim; j++) row[j] += scratch[j];
        }
    }

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> w, string key, long rows, long cols)
    {
        if (!w.TryGetValue(key, out Tensor? t)) throw new KeyNotFoundException($"The Drama 3 checkpoint has no tower weight '{key}'.");
        if (t.Shape.Rank != 2 || t.Shape[0] != rows || t.Shape[1] != cols)
        {
            throw new InvalidDataException($"Drama 3 tower weight '{key}' is {t.Shape}; expected [{rows}, {cols}].");
        }
        return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (TowerLayer layer in _shared) layer.Dispose();
        foreach (TowerLayer layer in _residual) layer.Dispose();
    }

    /// <summary>One tower layer: a code embedding table, the mix projection and its norms, one transformer layer and its cache.</summary>
    private sealed class TowerLayer(TransformerConfig block, FishAudioDrama3TowerConfig tower) : IDisposable
    {
        private readonly GenericTransformer _block = new(block);
        private readonly List<Tensor> _owned = [];
        private readonly float[] _row = new float[tower.Dim];
        private Tensor _embedding = null!, _enorm = null!, _hnorm = null!, _ehProj = null!;
        private IKvCache? _cache;

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix, string embeddingName)
        {
            int dim = tower.Dim, qWidth = tower.NumHeads * tower.HeadDim, kvWidth = tower.NumKvHeads * tower.HeadDim;
            _embedding = Require(w, $"{prefix}.{embeddingName}.weight", tower.VocabSize, dim);
            _enorm = Own(NormWeights.OnePlus(Require1(w, $"{prefix}.enorm.weight", dim)));
            _hnorm = Own(NormWeights.OnePlus(Require1(w, $"{prefix}.hnorm.weight", dim)));
            _ehProj = Require(w, $"{prefix}.eh_proj.weight", dim, 2L * dim);
            string t = $"{prefix}.transformer_layer";
            Tensor wqkv = Require(w, $"{t}.attention.wqkv.weight", 2L * qWidth + 2L * kvWidth, dim);
            Dictionary<string, Tensor> layer = new(StringComparer.Ordinal)
            {
                ["tower.layers.0.self_attn.q_proj.weight"] = wqkv.SliceRows(0, qWidth),
                ["tower.layers.0.self_attn.gate_proj.weight"] = wqkv.SliceRows(qWidth, qWidth),
                ["tower.layers.0.self_attn.k_proj.weight"] = wqkv.SliceRows(2L * qWidth, kvWidth),
                ["tower.layers.0.self_attn.v_proj.weight"] = wqkv.SliceRows(2L * qWidth + kvWidth, kvWidth),
                ["tower.layers.0.self_attn.o_proj.weight"] = Require(w, $"{t}.attention.wo.weight", dim, qWidth),
                // Always on and zero-centered whatever audio_decoder_config.attention_qk_norm says (see fish-speech TowerAttention).
                ["tower.layers.0.self_attn.q_norm.weight"] = Require1(w, $"{t}.attention.q_norm.weight", tower.HeadDim),
                ["tower.layers.0.self_attn.k_norm.weight"] = Require1(w, $"{t}.attention.k_norm.weight", tower.HeadDim),
                ["tower.layers.0.input_layernorm.weight"] = Require1(w, $"{t}.attention_norm.weight", dim),
                ["tower.layers.0.post_attention_layernorm.weight"] = Require1(w, $"{t}.ffn_norm.weight", dim),
                ["tower.layers.0.mlp.gate_proj.weight"] = Require(w, $"{t}.feed_forward.w1.weight", tower.IntermediateSize, dim),
                ["tower.layers.0.mlp.up_proj.weight"] = Require(w, $"{t}.feed_forward.w3.weight", tower.IntermediateSize, dim),
                ["tower.layers.0.mlp.down_proj.weight"] = Require(w, $"{t}.feed_forward.w2.weight", dim, tower.IntermediateSize),
                ["tower.norm.weight"] = Require1(w, $"{prefix}.final_layernorm.weight", dim),
            };
            _block.LoadWeightsHeadless(layer, "tower");
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            yield return _enorm;
            yield return _hnorm;
            yield return _ehProj;
            foreach (Tensor t in _block.EnumerateWeights()) yield return t;
        }

        public void BeginSequence(int capacity)
        {
            if (_cache is ISequenceState state && state.Capacity >= capacity)
            {
                state.Truncate(0);
                return;
            }
            _cache?.Dispose();
            _cache = KvCaches.ForDecode(1, tower.NumKvHeads, tower.HeadDim, capacity);
        }

        // e = semantic ? embedding(code) : 0, plus the shared input; then final_norm(block(eh_proj([enorm(e); hnorm(h)]))).
        public Tensor Forward(IBackend backend, Tensor hidden, int[] codes, bool[] semantic, int position, int count, Tensor? sharedInput)
        {
            IKvCache cache = _cache ?? throw new InvalidOperationException("BeginSequence must run before the tower.");
            int dim = tower.Dim;
            TensorShape rows = new(1, count, dim);
            using Tensor e = new(rows, DType.F32);
            float* ep = (float*)e.DataPointer;
            if (sharedInput is null) new Span<float>(ep, count * dim).Clear();
            else Buffer.MemoryCopy((void*)sharedInput.DataPointer, ep, (long)count * dim * 4, (long)count * dim * 4);
            fixed (float* row = _row)
            {
                for (int r = 0; r < count; r++)
                {
                    if (!semantic[r]) continue;
                    Qwen35Model.CopyEmbeddingRow(_embedding, Math.Clamp(codes[r], 0, tower.VocabSize - 1), dim, row);
                    float* dst = ep + (long)r * dim;
                    for (int j = 0; j < dim; j++) dst[j] += row[j];
                }
            }
            using Tensor en = new(rows, DType.F32);
            backend.RmsNorm(en, e, _enorm, tower.NormEps);
            using Tensor hn = new(rows, DType.F32);
            backend.RmsNorm(hn, hidden, _hnorm, tower.NormEps);
            using Tensor joined = new(new TensorShape(1, count, 2 * dim), DType.F32);
            backend.Concat(joined, [en, hn], 2);
            using Tensor mixed = new(rows, DType.F32);
            backend.Linear(mixed, joined, _ehProj, null);
            return _block.ForwardEmbeds(backend, mixed, count, position, cache, applyFinalNorm: true);
        }

        private Tensor Own(Tensor t)
        {
            _owned.Add(t);
            return t;
        }

        private static Tensor Require1(IReadOnlyDictionary<string, Tensor> w, string key, int length)
        {
            if (!w.TryGetValue(key, out Tensor? t)) throw new KeyNotFoundException($"The Drama 3 checkpoint has no tower weight '{key}'.");
            if (t.ElementCount != length) throw new InvalidDataException($"Drama 3 tower weight '{key}' is {t.Shape}; expected [{length}].");
            return t;
        }

        public void Dispose()
        {
            _cache?.Dispose();
            _block.Dispose();
            foreach (Tensor t in _owned) t.Dispose();
        }
    }
}
