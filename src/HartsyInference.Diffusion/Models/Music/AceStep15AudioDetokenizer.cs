using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Models.Music;

/// <summary>ACE-Step v1.5 audio-code detokenizer: 5 Hz FSQ indices → 25 Hz acoustic latents <c>[1, T·5, 64]</c>
/// (the LM planner's <c>lmHints</c>). Mirrors upstream <c>ResidualFSQ.get_output_from_indices</c> (levels
/// [8,8,8,5,5,5], 1 quantizer, scale 1) + <c>AudioTokenDetokenizer</c> (embed Linear → ×5 patch expansion with
/// learnable special tokens → 2 bidirectional Qwen3 encoder layers per 5-patch group → norm → proj_out).
/// Weights come from the DiT checkpoint's retained <c>tokenizer.quantizer.*</c> / <c>detokenizer.*</c> keys.</summary>
public sealed unsafe class AceStep15AudioDetokenizer : IDisposable
{
    private static readonly int[] FsqLevels = [8, 8, 8, 5, 5, 5];
    /// <summary>Highest valid FSQ index (∏ levels − 1 = 63,999).</summary>
    public const int MaxCode = 8 * 8 * 8 * 5 * 5 * 5 - 1;

    private readonly AceStep15Config _c;
    private readonly AceStep15EncoderLayer[] _layers;
    private Tensor? _fsqOutW, _fsqOutB;          // quantizer.project_out [2048, 6]
    private Tensor? _embedW, _embedB;            // detokenizer.embed_tokens [2048, 2048]
    private Tensor? _special;                    // detokenizer.special_tokens [1, 5, 2048]
    private Tensor? _norm;                       // detokenizer.norm
    private Tensor? _projW, _projB;              // detokenizer.proj_out [64, 2048]
    private int _disposed;

    /// <summary>Patches per token (5 Hz → 25 Hz).</summary>
    public int PoolWindow { get; }

    public AceStep15AudioDetokenizer(AceStep15Config config, int poolWindow = 5, int poolerLayers = 2)
    {
        _c = config.EncoderVariant();
        PoolWindow = poolWindow;
        _layers = new AceStep15EncoderLayer[poolerLayers];
        for (int i = 0; i < poolerLayers; i++) _layers[i] = new AceStep15EncoderLayer(_c);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _fsqOutW = w["tokenizer.quantizer.project_out.weight"];
        _fsqOutB = w["tokenizer.quantizer.project_out.bias"];
        _embedW = w["detokenizer.embed_tokens.weight"];
        _embedB = w["detokenizer.embed_tokens.bias"];
        _special = TensorCasts.EnsureF32(w["detokenizer.special_tokens"]);
        _norm = TensorCasts.EnsureF32(w["detokenizer.norm.weight"]);
        _projW = w["detokenizer.proj_out.weight"];
        _projB = w["detokenizer.proj_out.bias"];
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(w, $"detokenizer.layers.{i}");
    }

    /// <summary>Decodes 5 Hz FSQ code indices (0..<see cref="MaxCode"/>, clamped) into 25 Hz latents
    /// <c>[1, codes.Length·5, 64]</c>.</summary>
    public Tensor Decode(IBackend backend, ReadOnlySpan<int> codes)
    {
        ThrowIfDisposed();
        if (_fsqOutW is null) throw new InvalidOperationException("Detokenizer weights not loaded.");
        int t = codes.Length;
        int hidden = _c.HiddenSize;
        int p = PoolWindow;

        // FSQ dequant: digit_i = (index / basis_i) % level_i; value = (digit − half)/half, half = (level−1)/2;
        // then project_out Linear(6 → 2048).
        Tensor fsqVals = new(new TensorShape(1, t, FsqLevels.Length), DType.F32);
        float* fv = (float*)fsqVals.DataPointer;
        for (int i = 0; i < t; i++)
        {
            int idx = Math.Clamp(codes[i], 0, MaxCode);
            int basis = 1;
            for (int d = 0; d < FsqLevels.Length; d++)
            {
                int level = FsqLevels[d];
                int digit = idx / basis % level;
                float half = (level - 1) / 2f;
                fv[(long)i * FsqLevels.Length + d] = (digit - half) / half;
                basis *= level;
            }
        }
        Tensor quantized = new(new TensorShape(1, t, hidden), DType.F32);
        backend.Linear(quantized, fsqVals, _fsqOutW!, _fsqOutB);
        fsqVals.Dispose();

        // embed_tokens Linear, then expand each token into P patches + learnable special tokens.
        Tensor embedded = new(new TensorShape(1, t, hidden), DType.F32);
        backend.Linear(embedded, quantized, _embedW!, _embedB);
        quantized.Dispose();
        Tensor patches = new(new TensorShape(1, (long)t * p, hidden), DType.F32);
        float* pp = (float*)patches.DataPointer;
        float* ep = (float*)embedded.DataPointer;
        float* sp = (float*)_special!.DataPointer;
        for (int i = 0; i < t; i++)
            for (int k = 0; k < p; k++)
            {
                long dst = ((long)i * p + k) * hidden;
                long src = (long)i * hidden;
                long spo = (long)k * hidden;
                for (int c = 0; c < hidden; c++) pp[dst + c] = ep[src + c] + sp[spo + c];
            }
        embedded.Dispose();

        // Per-group bidirectional encoding: each 5-patch group attends only within itself (upstream reshapes to
        // (b·t, 5, h)); RoPE positions 0..4. Window 128 ≫ 5, so sliding == full here.
        (float[] cos, float[] sin) = AceStep15Attention.BuildRopeTables(p, _c.HeadDim, _c.RopeTheta);
        Tensor outLat = new(new TensorShape(1, (long)t * p, _c.LatentChannels), DType.F32);
        float* op = (float*)outLat.DataPointer;
        for (int i = 0; i < t; i++)
        {
            Tensor group = new(new TensorShape(1, p, hidden), DType.F32);
            Buffer.MemoryCopy(pp + (long)i * p * hidden, (void*)group.DataPointer, (long)p * hidden * 4, (long)p * hidden * 4);
            Tensor cur = group;
            for (int li = 0; li < _layers.Length; li++)
            {
                Tensor next = _layers[li].Forward(backend, cur, cos, sin, mask: null);
                cur.Dispose();
                cur = next;
            }
            Tensor normed = new(cur.Shape, DType.F32);
            backend.RmsNorm(normed, cur, _norm!, _c.RmsNormEps);
            cur.Dispose();
            Tensor proj = new(new TensorShape(1, p, _c.LatentChannels), DType.F32);
            backend.Linear(proj, normed, _projW!, _projB);
            normed.Dispose();
            Buffer.MemoryCopy((void*)proj.DataPointer, op + (long)i * p * _c.LatentChannels,
                (long)p * _c.LatentChannels * 4, (long)p * _c.LatentChannels * 4);
            proj.Dispose();
        }
        patches.Dispose();
        return outLat;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _fsqOutW, _fsqOutB, _embedW, _embedB, _special, _norm, _projW, _projB })
            if (t is not null) yield return t;
        foreach (AceStep15EncoderLayer l in _layers)
            foreach (Tensor t in l.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(AceStep15AudioDetokenizer));
    }
}
