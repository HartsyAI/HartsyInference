using HartsyInference.Audio.Models.FishAudio;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>ControlFoley's CLIP conditioner (<c>FeaturesUtils.encode_text</c> / <c>encode_video_with_clip</c>): the open_clip
/// text tower returning L2-normalised last hidden states after <c>ln_final</c> (no pooling, no projection), and the
/// vision tower returning L2-normalised projected CLS embeddings of mean/std-normalised frames.</summary>
public sealed unsafe class ControlFoleyClip
{
    private static readonly float[] Mean = [0.48145466f, 0.4578275f, 0.40821073f];
    private static readonly float[] Std = [0.26862954f, 0.26130258f, 0.27577711f];

    private readonly ControlFoleyClipConfig _config;
    private readonly Lazy<ControlFoleyClipTokenizer> _tokenizer = new(() => new ControlFoleyClipTokenizer());
    private ControlFoleyClipTransformer? _text;
    private ControlFoleyClipTransformer? _vision;
    private Tensor? _tokenEmbedding, _textPositions, _lnFinalW, _lnFinalB;
    private Tensor? _patchW, _classEmbedding, _visionPositions, _lnPreW, _lnPreB, _lnPostW, _lnPostB, _projT;

    /// <summary>Creates an unloaded conditioner; call <see cref="LoadWeights"/> before encoding.</summary>
    public ControlFoleyClip(ControlFoleyClipConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
    }

    /// <summary>Geometry this instance was built for.</summary>
    public ControlFoleyClipConfig Config => _config;

    /// <summary>Binds an open_clip state dict (<c>token_embedding.weight</c>, <c>transformer.resblocks.*</c>, <c>visual.*</c>, etc.,
    /// optionally under <paramref name="prefix"/>) after validating every tensor shape. <c>text_projection</c> and <c>logit_scale</c> are unused.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "")
    {
        ArgumentNullException.ThrowIfNull(weights);
        string p = prefix.Length == 0 ? "" : prefix + ".";
        IReadOnlyDictionary<string, Tensor> w = prefix.Length == 0 ? weights : new PrefixView(weights, p);
        ControlFoleyClipConfig c = _config;
        int tokens = c.GridSize * c.GridSize + 1;

        _tokenEmbedding = ControlFoleyClipTransformer.Load(w, "token_embedding.weight", c.VocabSize, c.TextWidth);
        _textPositions = ControlFoleyClipTransformer.Load(w, "positional_embedding", c.ContextLength, c.TextWidth);
        _lnFinalW = ControlFoleyClipTransformer.Load(w, "ln_final.weight", c.TextWidth);
        _lnFinalB = ControlFoleyClipTransformer.Load(w, "ln_final.bias", c.TextWidth);
        _text = new ControlFoleyClipTransformer(w, "transformer", c.TextLayers, c.TextWidth, c.TextHeads, c.MlpRatio, c.LayerNormEps);

        int patchDim = 3 * c.PatchSize * c.PatchSize;
        Tensor conv = ControlFoleyClipTransformer.Load(w, "visual.conv1.weight", c.VisionWidth, 3, c.PatchSize, c.PatchSize);
        _patchW = new Tensor(new TensorShape(c.VisionWidth, patchDim), DType.F32);
        new ReadOnlySpan<float>((void*)conv.DataPointer, c.VisionWidth * patchDim)
            .CopyTo(new Span<float>((void*)_patchW.DataPointer, c.VisionWidth * patchDim));
        _classEmbedding = ControlFoleyClipTransformer.Load(w, "visual.class_embedding", c.VisionWidth);
        _visionPositions = ControlFoleyClipTransformer.Load(w, "visual.positional_embedding", tokens, c.VisionWidth);
        _lnPreW = ControlFoleyClipTransformer.Load(w, "visual.ln_pre.weight", c.VisionWidth);
        _lnPreB = ControlFoleyClipTransformer.Load(w, "visual.ln_pre.bias", c.VisionWidth);
        _lnPostW = ControlFoleyClipTransformer.Load(w, "visual.ln_post.weight", c.VisionWidth);
        _lnPostB = ControlFoleyClipTransformer.Load(w, "visual.ln_post.bias", c.VisionWidth);
        Tensor proj = ControlFoleyClipTransformer.Load(w, "visual.proj", c.VisionWidth, c.EmbedDim);
        _projT = new Tensor(new TensorShape(c.EmbedDim, c.VisionWidth), DType.F32);
        float* src = (float*)proj.DataPointer, dst = (float*)_projT.DataPointer;
        for (int i = 0; i < c.VisionWidth; i++)
        {
            for (int j = 0; j < c.EmbedDim; j++)
            {
                dst[j * c.VisionWidth + i] = src[i * c.EmbedDim + j];
            }
        }

        _vision = new ControlFoleyClipTransformer(w, "visual.transformer", c.VisionLayers, c.VisionWidth, c.VisionHeads, c.MlpRatio,
            c.LayerNormEps);
    }

    /// <summary>Tokenizes with the open_clip tokenizer and encodes; returns <c>[N, ContextLength, TextWidth]</c> L2-normalised over the last dimension.</summary>
    public float[] EncodeText(IBackend backend, IReadOnlyList<string> prompts)
    {
        ArgumentNullException.ThrowIfNull(prompts);
        return EncodeTokens(backend, _tokenizer.Value.EncodeBatch(prompts));
    }

    /// <summary>Encodes pre-tokenized prompts (each exactly <see cref="ControlFoleyClipConfig.ContextLength"/> ids).</summary>
    public float[] EncodeTokens(IBackend backend, IReadOnlyList<int[]> tokens)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(tokens);
        if (_text is null)
        {
            throw new InvalidOperationException("Call LoadWeights before encoding.");
        }

        int n = _config.ContextLength, w = _config.TextWidth;
        float[] result = new float[tokens.Count * n * w];
        float* emb = (float*)_tokenEmbedding!.DataPointer, pos = (float*)_textPositions!.DataPointer;
        for (int b = 0; b < tokens.Count; b++)
        {
            if (tokens[b].Length != n)
            {
                throw new ArgumentException($"Prompt {b} has {tokens[b].Length} tokens, expected {n}.", nameof(tokens));
            }

            float[] x = new float[n * w];
            for (int t = 0; t < n; t++)
            {
                int id = tokens[b][t];
                if ((uint)id >= (uint)_config.VocabSize)
                {
                    throw new ArgumentOutOfRangeException(nameof(tokens), $"Token id {id} is outside the {_config.VocabSize}-entry vocabulary.");
                }

                for (int d = 0; d < w; d++)
                {
                    x[t * w + d] = emb[(long)id * w + d] + pos[t * w + d];
                }
            }

            x = _text.Forward(backend, x, n, causal: true);
            x = ControlFoleyClipTransformer.LayerNorm(backend, x, _lnFinalW!, _lnFinalB!, n, w, _config.LayerNormEps);
            for (int t = 0; t < n; t++)
            {
                Normalize(x.AsSpan(t * w, w));
            }

            x.CopyTo(result, b * n * w);
        }

        return result;
    }

    /// <summary>Encodes frames <c>[T, 3, S, S]</c> (S = <see cref="ControlFoleyClipConfig.InputSize"/>, values 0..1) to <c>[T, EmbedDim]</c> L2-normalised CLS embeddings.</summary>
    public float[] EncodeImages(IBackend backend, float[] frames)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(frames);
        if (_vision is null)
        {
            throw new InvalidOperationException("Call LoadWeights before encoding.");
        }

        ControlFoleyClipConfig c = _config;
        int size = c.InputSize, plane = size * size, perFrame = 3 * plane;
        if (frames.Length == 0 || frames.Length % perFrame != 0)
        {
            throw new ArgumentException($"Expected a multiple of 3x{size}x{size} floats, got {frames.Length}.", nameof(frames));
        }

        int count = frames.Length / perFrame, grid = c.GridSize, p = c.PatchSize, w = c.VisionWidth;
        int tokens = grid * grid + 1, patchDim = 3 * p * p;
        float[] result = new float[count * c.EmbedDim];
        float* cls = (float*)_classEmbedding!.DataPointer, pos = (float*)_visionPositions!.DataPointer;
        for (int f = 0; f < count; f++)
        {
            float[] patches = new float[grid * grid * patchDim];
            for (int gy = 0; gy < grid; gy++)
            {
                for (int gx = 0; gx < grid; gx++)
                {
                    int row = (gy * grid + gx) * patchDim;
                    for (int ch = 0; ch < 3; ch++)
                    {
                        for (int ky = 0; ky < p; ky++)
                        {
                            for (int kx = 0; kx < p; kx++)
                            {
                                float v = frames[f * perFrame + ch * plane + (gy * p + ky) * size + gx * p + kx];
                                patches[row + (ch * p + ky) * p + kx] = (v - Mean[ch]) / Std[ch];
                            }
                        }
                    }
                }
            }

            float[] embedded = DacOps.Linear(backend, patches, _patchW!, grid * grid, patchDim, w);
            float[] x = new float[tokens * w];
            for (int d = 0; d < w; d++)
            {
                x[d] = cls[d] + pos[d];
            }

            for (int t = 1; t < tokens; t++)
            {
                for (int d = 0; d < w; d++)
                {
                    x[t * w + d] = embedded[(t - 1) * w + d] + pos[t * w + d];
                }
            }

            x = ControlFoleyClipTransformer.LayerNorm(backend, x, _lnPreW!, _lnPreB!, tokens, w, c.LayerNormEps);
            x = _vision.Forward(backend, x, tokens, causal: false);
            float[] pooled = ControlFoleyClipTransformer.LayerNorm(backend, x[..w], _lnPostW!, _lnPostB!, 1, w, c.LayerNormEps);
            float[] embedding = DacOps.Linear(backend, pooled, _projT!, 1, w, c.EmbedDim);
            Normalize(embedding);
            embedding.CopyTo(result, f * c.EmbedDim);
        }

        return result;
    }

    /// <summary>torch <c>F.normalize(x, dim=-1)</c> with eps 1e-12.</summary>
    private static void Normalize(Span<float> row)
    {
        double sum = 0;
        foreach (float v in row)
        {
            sum += (double)v * v;
        }

        float inv = 1f / MathF.Max((float)Math.Sqrt(sum), 1e-12f);
        for (int i = 0; i < row.Length; i++)
        {
            row[i] *= inv;
        }
    }

    private sealed class PrefixView(IReadOnlyDictionary<string, Tensor> inner, string prefix) : IReadOnlyDictionary<string, Tensor>
    {
        public Tensor this[string key] => inner[prefix + key];

        public IEnumerable<string> Keys => inner.Keys.Where(k => k.StartsWith(prefix, StringComparison.Ordinal)).Select(k => k[prefix.Length..]);

        public IEnumerable<Tensor> Values => Keys.Select(k => this[k]);

        public int Count => Keys.Count();

        public bool ContainsKey(string key) => inner.ContainsKey(prefix + key);

        public bool TryGetValue(string key, out Tensor value) => inner.TryGetValue(prefix + key, out value!);

        public IEnumerator<KeyValuePair<string, Tensor>> GetEnumerator() =>
            Keys.Select(k => new KeyValuePair<string, Tensor>(k, this[k])).GetEnumerator();

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
