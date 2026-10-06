using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.BreezeTts;
using HartsyInference.Audio.Models.Codecs.Mimi;
using HartsyInference.Audio.Models.QwenTts;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Breeze TTS 2 text-to-speech: voice clone, voice design and voice direction. Builds the segment prompt, encodes
/// text segments with the T5Gemma2 encoder, runs the Qwen3 backbone + depth decoder frame by frame (optionally with
/// classifier-free guidance against the template's negative prompt), and decodes the 16-codebook frames with the
/// Qwen3-TTS-Tokenizer-12Hz vocoder. Port of <c>infer.py</c> / <c>fast_streaming.py</c>'s eager semantics.</summary>
public sealed unsafe class BreezeTts2Pipeline : IDisposable
{
    private readonly BreezeTts2Config _cfg;
    private readonly SentencePieceBpeJson _tokenizer;
    private readonly T5Gemma2TextEncoder _textEncoder;
    private readonly BreezeTts2Model _model;
    private readonly Qwen3TtsVocoder _vocoder;
    private readonly Mimi _referenceEncoder;
    private Tensor? _textProj;
    private int _disposed;

    public BreezeTts2Pipeline(BreezeTts2Config cfg, SentencePieceBpeJson tokenizer)
    {
        _cfg = cfg;
        _tokenizer = tokenizer;
        _textEncoder = new T5Gemma2TextEncoder(cfg.TextEncoder);
        _model = new BreezeTts2Model(cfg);
        _vocoder = new Qwen3TtsVocoder(new Qwen3TtsVocoderConfig());
        _referenceEncoder = new Mimi(cfg.Codec);
    }

    public int SampleRate => _vocoder.SampleRate;

    /// <param name="model">The main checkpoint (<c>backbone_model.*</c>, <c>depth_decoder.*</c>, <c>text_encoder*.*</c>, <c>lm_head.weight</c>).</param>
    /// <param name="audioTokenizer">The bundled <c>audio_tokenizer/model.safetensors</c> (<c>decoder.*</c> vocoder, <c>encoder.*</c> Mimi encoder).</param>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> model, IReadOnlyDictionary<string, Tensor> audioTokenizer)
    {
        _model.LoadWeights(model);
        _textEncoder.LoadWeights(model);
        _textProj = model["text_encoder_proj.weight"];
        _vocoder.LoadWeights(audioTokenizer);

        // The tokenizer's Mimi encoder is HF-layout with an extra leading "encoder." on every key.
        Dictionary<string, Tensor> mimi = new();
        foreach (KeyValuePair<string, Tensor> kv in audioTokenizer)
            if (kv.Key.StartsWith("encoder.", StringComparison.Ordinal)) mimi[kv.Key["encoder.".Length..]] = kv.Value;
        _referenceEncoder.LoadWeights(mimi);
    }

    /// <summary>Encodes a 24 kHz mono reference clip into <c>[T][16]</c> frames (Mimi, first 16 quantizers).</summary>
    public int[][] EncodeReference(IBackend backend, ReadOnlySpan<float> pcm24k)
    {
        const int Hop = 1_920;
        int frames = (pcm24k.Length + Hop - 1) / Hop;
        int padded = frames * Hop;
        using Tensor pcm = new(new TensorShape(1, 1, padded), DType.F32);
        pcm24k.CopyTo(new Span<float>((void*)pcm.DataPointer, pcm24k.Length));
        using Tensor codes = _referenceEncoder.Encode(backend, pcm, 1, padded);   // [1, K, T] Int32
        int k = (int)codes.Shape[1], t = Math.Min(frames, (int)codes.Shape[2]), n = _cfg.NumCodebooks;
        int* p = (int*)codes.DataPointer;
        int[][] result = new int[t][];
        for (int j = 0; j < t; j++)
        {
            result[j] = new int[n];
            for (int i = 0; i < n; i++) result[j][i] = p[(long)i * (int)codes.Shape[2] + j];
        }
        return result;
    }

    public sealed record Request
    {
        public required string Text { get; init; }
        public string? Instruction { get; init; }
        public int[][]? ReferenceFrames { get; init; }
        public string? ReferenceText { get; init; }
        public string Speaker { get; init; } = "S0";
        /// <summary>1 = no guidance. &gt; 1 strengthens the instruction (design / direction). Needs a template with a negative prompt.</summary>
        public float CfgScale { get; init; } = 1f;
        public float? Temperature { get; init; }
        public int? TopK { get; init; }
        public float? TopP { get; init; }
        public float? RepetitionPenalty { get; init; }
        public int? MaxFrames { get; init; }
        public int Seed { get; init; }
        public CancellationToken Cancel { get; init; }
    }

    /// <summary>Generates and decodes; returns 24 kHz mono PCM (empty when no frame was produced).</summary>
    public float[] Synthesize(IBackend backend, Request req)
    {
        int[,] grid = GenerateCodes(backend, req);
        return grid.GetLength(1) == 0 ? [] : _vocoder.Decode(backend, grid);
    }

    /// <summary>Runs generation only; returns the <c>[numCodebooks, T]</c> code grid.</summary>
    public int[,] GenerateCodes(IBackend backend, Request req)
    {
        ThrowIfDisposed();
        BreezeTts2Templates.Request tr = new()
        {
            Text = req.Text, Instruction = req.Instruction, RefFrames = req.ReferenceFrames, RefText = req.ReferenceText,
            Speaker = req.Speaker,
        };
        List<BreezeSegment> positive = BreezeTts2Templates.Positive(tr);
        bool guided = req.CfgScale != 1f;
        List<BreezeSegment>? negative = guided ? BreezeTts2Templates.Negative(tr) : null;
        if (guided && negative is null)
            throw new ArgumentException("This request type has no negative prompt, so a guidance scale other than 1 is not supported.");

        float temperature = req.Temperature ?? _cfg.Temperature, topP = req.TopP ?? _cfg.TopP, rep = req.RepetitionPenalty ?? _cfg.RepetitionPenalty;
        int topK = req.TopK ?? _cfg.TopK, maxFrames = req.MaxFrames is > 0 ? req.MaxFrames.Value : _cfg.MaxNewFrames;
        int h = _cfg.HiddenSize, n = _cfg.NumCodebooks;

        float[] cond = BuildEmbeds(backend, positive, out int condLen);
        float[]? uncond = negative is null ? null : BuildEmbeds(backend, negative, out _);
        int uncondLen = uncond is null ? 0 : uncond.Length / h;
        maxFrames = Math.Min(maxFrames, Math.Max(1, _cfg.MaxSequenceLength - Math.Max(condLen, uncondLen) - 1));

        using IKvCache condCache = _model.CreateBackboneCache(condLen + maxFrames + 2);
        using IKvCache? uncondCache = uncond is null ? null : _model.CreateBackboneCache(uncondLen + maxFrames + 2);
        float[] condHidden = _model.BackboneStep(backend, cond, condLen, 0, condCache);
        float[]? uncondHidden = uncond is null ? null : _model.BackboneStep(backend, uncond, uncondLen, 0, uncondCache!);

        uint rng = DeterministicRng.Seed(req.Seed);
        List<int[]> frames = [];
        List<int> history = [];
        int condPos = condLen, uncondPos = uncondLen;
        float[] Logits(float[] hc, float[]? hu)
        {
            float[] c = _model.BackboneLogits(backend, hc);
            if (hu is null) return c;
            float[] u = _model.BackboneLogits(backend, hu);
            for (int i = 0; i < c.Length; i++) c[i] = u[i] + req.CfgScale * (c[i] - u[i]);
            return c;
        }

        int eos = _cfg.AudioVocabSize;   // the backbone head has one extra class after the codebook vocabulary
        bool first = true;
        float[] logits = Logits(condHidden, uncondHidden);
        while (frames.Count < maxFrames)
        {
            req.Cancel.ThrowIfCancellationRequested();
            if (!first) ApplyRepetitionPenalty(logits, history, rep);
            first = false;
            for (int r = _cfg.CodebookSize; r < _cfg.AudioVocabSize; r++) logits[r] = float.NegativeInfinity;
            int token = NucleusSampler.Draw(logits, logits.Length, temperature, topK, topP, ref rng);
            if (token == eos) break;
            history.Add(token);

            int[] frame = _model.SampleFrame(backend, condHidden, token, ref rng, uncondHidden, req.CfgScale, temperature, topK, topP);
            if (frame.All(c => c == _cfg.CodebookPadTokenId)) break;
            frames.Add(frame);

            float[] embed = _model.EmbedFrame(frame);
            condHidden = _model.BackboneStep(backend, embed, 1, condPos++, condCache);
            if (uncondHidden is not null) uncondHidden = _model.BackboneStep(backend, embed, 1, uncondPos++, uncondCache!);
            logits = Logits(condHidden, uncondHidden);
        }

        int[,] grid = new int[n, frames.Count];
        for (int j = 0; j < frames.Count; j++)
            for (int i = 0; i < n; i++) grid[i, j] = frames[j][i];
        return grid;
    }

    // HF RepetitionPenaltyLogitsProcessor over the tokens sampled so far: positive scores shrink, negative grow.
    private static void ApplyRepetitionPenalty(float[] logits, List<int> history, float penalty)
    {
        if (penalty == 1f) return;
        foreach (int t in history.Distinct())
            logits[t] = logits[t] > 0 ? logits[t] / penalty : logits[t] * penalty;
    }

    // Prompt embeddings: text segments through the text encoder + projection, audio frames through the audio embedding.
    private float[] BuildEmbeds(IBackend backend, List<BreezeSegment> segments, out int length)
    {
        int h = _cfg.HiddenSize;
        List<float[]> rows = [];
        foreach (BreezeSegment segment in segments)
        {
            if (segment is BreezeTextSegment text)
            {
                // each text segment gets its own <bos> and is encoded as an independent sequence
                int[] ids = [_tokenizer.BosId, .. _tokenizer.Encode(text.Text)];
                float[] hidden = _textEncoder.Encode(backend, ids);
                using Tensor input = new(new TensorShape(1, ids.Length, _cfg.TextEncoder.HiddenSize), DType.F32);
                hidden.AsSpan().CopyTo(new Span<float>((void*)input.DataPointer, hidden.Length));
                using Tensor projected = new(new TensorShape(1, ids.Length, h), DType.F32);
                backend.Linear(projected, input, _textProj!, null);
                float* p = (float*)projected.DataPointer;
                for (int i = 0; i < ids.Length; i++) rows.Add(new ReadOnlySpan<float>(p + (long)i * h, h).ToArray());
            }
            else if (segment is BreezeAudioSegment audio)
            {
                int[][] frames = audio.DropLastFrame ? audio.Frames[..^1] : audio.Frames;
                foreach (int[] f in frames) rows.Add(_model.EmbedFrame(f));
                if (audio.AppendEos)
                    rows.Add(_model.EmbedFrame(Enumerable.Repeat(_cfg.CodebookEosTokenId, _cfg.NumCodebooks).ToArray()));
            }
        }
        length = rows.Count;
        float[] flat = new float[length * h];
        for (int i = 0; i < length; i++) rows[i].CopyTo(flat, i * h);
        return flat;
    }

    public IEnumerable<Tensor> EnumerateWeights() => _model.EnumerateWeights();

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _textEncoder.Dispose(); _model.Dispose(); _vocoder.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(BreezeTts2Pipeline));
    }
}
