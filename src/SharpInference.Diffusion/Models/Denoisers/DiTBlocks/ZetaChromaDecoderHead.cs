using SharpInference.Core.Backends;
using SharpInference.Core.Exceptions;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Zeta-Chroma pixel decoder head — ComfyUI's <c>SimpleMLPAdaLN</c> (<c>dec_net.*</c>), the DeCo-style
/// replacement for Z-Image's <c>final_layer</c>. Each image patch becomes ONE decoder token: the whole flattened
/// noisy patch (patch²·3 values) plus maxFreqs² constant DCT features (the shared NerfEmbedder runs with an
/// internal patch size of 1, so its cosine basis collapses to <c>1/(1+u·v)</c>) is projected to the decoder width,
/// then refined by AdaLN residual blocks conditioned on the backbone's per-patch hidden state:
/// <list type="number">
///   <item><c>dec_net.input_embedder.embedder.0</c>: Linear(patch²·3 + maxFreqs² → C).</item>
///   <item><c>dec_net.cond_embed</c>: Linear(backboneHidden → C), one conditioning vector per patch.</item>
///   <item><c>dec_net.res_blocks.{i}</c>: <c>shift, scale, gate = chunk3(adaLN_modulation(silu(y)))</c>;
///         <c>h += gate ⊙ mlp(ln(h)·(1+scale) + shift)</c> with affine <c>in_ln</c> (eps 1e-6) and a
///         Linear→SiLU→Linear MLP.</item>
///   <item><c>dec_net.final_layer</c>: LayerNorm-no-affine (eps 1e-6) → Linear(C → patch²·3).</item>
/// </list>
/// Everything here is validation-gated (the model is mid-pretraining; key layout verified only against ComfyUI
/// <c>comfy/ldm/lumina/model.py</c>, not a checkpoint dump) — hence the defensive <see cref="LoadWeights"/> that
/// throws <see cref="UnsupportedModelException"/> listing every missing/unmatched <c>dec_net.*</c> key.</summary>
public sealed unsafe class ZetaChromaDecoderHead : IDisposable
{
    private const float LnEps = 1e-6f;

    private readonly int _patchDim;
    private readonly int _maxFreqs;
    private readonly int _resBlockCount;

    private Tensor? _condWeight, _condBias;
    private Tensor? _embedWeight, _embedBias;
    private readonly Tensor?[] _lnWeight;
    private readonly Tensor?[] _lnBias;
    private readonly Tensor?[] _mlp0Weight, _mlp0Bias;
    private readonly Tensor?[] _mlp2Weight, _mlp2Bias;
    private readonly Tensor?[] _adaWeight, _adaBias;
    private Tensor? _finalWeight, _finalBias;
    private int _channels;
    private int _disposed;

    /// <summary>Creates the head from config dims (weights loaded separately).</summary>
    /// <param name="patchDim">Flattened patch size: pixel patch² × 3.</param>
    /// <param name="maxFreqs">DCT frequency count per axis (8 → 64 constant features).</param>
    /// <param name="resBlockCount">Number of AdaLN residual blocks (4).</param>
    public ZetaChromaDecoderHead(int patchDim, int maxFreqs, int resBlockCount)
    {
        if (patchDim <= 0 || maxFreqs <= 0 || resBlockCount <= 0)
            throw new ArgumentException($"Invalid decoder head dims: patchDim={patchDim}, maxFreqs={maxFreqs}, resBlocks={resBlockCount}.");
        _patchDim = patchDim;
        _maxFreqs = maxFreqs;
        _resBlockCount = resBlockCount;
        _lnWeight = new Tensor?[resBlockCount];
        _lnBias = new Tensor?[resBlockCount];
        _mlp0Weight = new Tensor?[resBlockCount];
        _mlp0Bias = new Tensor?[resBlockCount];
        _mlp2Weight = new Tensor?[resBlockCount];
        _mlp2Bias = new Tensor?[resBlockCount];
        _adaWeight = new Tensor?[resBlockCount];
        _adaBias = new Tensor?[resBlockCount];
    }

    /// <summary>Decoder hidden width, read from <c>dec_net.cond_embed.weight</c> on load.</summary>
    public int Channels => _channels;

    /// <summary>Loads all <c>dec_net.*</c> weights defensively: any missing required key OR any <c>dec_net.*</c>
    /// key left unconsumed means the checkpoint's decoder layout differs from the assumed SimpleMLPAdaLN —
    /// throws <see cref="UnsupportedModelException"/> listing the offending keys so the dump can be diffed.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        List<string> missing = new();
        HashSet<string> consumed = new(StringComparer.Ordinal);

        Tensor? Required(string key)
        {
            if (weights.TryGetValue(key, out Tensor? t))
            {
                consumed.Add(key);
                return t;
            }
            missing.Add(key);
            return null;
        }

        Tensor? Optional(string key)
        {
            if (weights.TryGetValue(key, out Tensor? t))
            {
                consumed.Add(key);
                return t;
            }
            return null;
        }

        _condWeight = Required("dec_net.cond_embed.weight");
        _condBias = Optional("dec_net.cond_embed.bias");
        _embedWeight = Required("dec_net.input_embedder.embedder.0.weight");
        _embedBias = Optional("dec_net.input_embedder.embedder.0.bias");

        for (int i = 0; i < _resBlockCount; i++)
        {
            string b = $"dec_net.res_blocks.{i}";
            _lnWeight[i] = Required($"{b}.in_ln.weight");
            _lnBias[i] = Optional($"{b}.in_ln.bias");
            _mlp0Weight[i] = Required($"{b}.mlp.0.weight");
            _mlp0Bias[i] = Optional($"{b}.mlp.0.bias");
            _mlp2Weight[i] = Required($"{b}.mlp.2.weight");
            _mlp2Bias[i] = Optional($"{b}.mlp.2.bias");
            _adaWeight[i] = Required($"{b}.adaLN_modulation.1.weight");
            _adaBias[i] = Optional($"{b}.adaLN_modulation.1.bias");
        }

        // final_layer.norm_final is elementwise_affine=False — no checkpoint keys.
        _finalWeight = Required("dec_net.final_layer.linear.weight");
        _finalBias = Optional("dec_net.final_layer.linear.bias");

        List<string> unmatched = new();
        foreach (string key in weights.Keys)
        {
            if (key.StartsWith("dec_net.", StringComparison.Ordinal) && !consumed.Contains(key))
                unmatched.Add(key);
        }

        if (missing.Count > 0 || unmatched.Count > 0)
        {
            unmatched.Sort(StringComparer.Ordinal);
            throw new UnsupportedModelException(
                "Zeta-Chroma decoder head layout mismatch (the assumed SimpleMLPAdaLN structure is " +
                "validation-gated — dump these keys and update ZetaChromaDecoderHead). " +
                $"Missing: [{string.Join(", ", missing)}]; unmatched dec_net keys: [{string.Join(", ", unmatched)}]",
                architecture: "zeta-chroma", format: null);
        }

        _channels = (int)_condWeight!.Shape[0];
        long expectedEmbedIn = _patchDim + (long)_maxFreqs * _maxFreqs;
        if (_embedWeight!.Shape[1] != expectedEmbedIn)
        {
            throw new UnsupportedModelException(
                $"dec_net.input_embedder.embedder.0.weight in-dim {_embedWeight.Shape[1]} != expected " +
                $"{expectedEmbedIn} (patchDim {_patchDim} + maxFreqs² {_maxFreqs * _maxFreqs}).",
                architecture: "zeta-chroma", format: null);
        }
    }

    /// <summary>Enumerates weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_condWeight is not null) yield return _condWeight;
        if (_condBias is not null) yield return _condBias;
        if (_embedWeight is not null) yield return _embedWeight;
        if (_embedBias is not null) yield return _embedBias;
        for (int i = 0; i < _resBlockCount; i++)
        {
            if (_lnWeight[i] is not null) yield return _lnWeight[i]!;
            if (_lnBias[i] is not null) yield return _lnBias[i]!;
            if (_mlp0Weight[i] is not null) yield return _mlp0Weight[i]!;
            if (_mlp0Bias[i] is not null) yield return _mlp0Bias[i]!;
            if (_mlp2Weight[i] is not null) yield return _mlp2Weight[i]!;
            if (_mlp2Bias[i] is not null) yield return _mlp2Bias[i]!;
            if (_adaWeight[i] is not null) yield return _adaWeight[i]!;
            if (_adaBias[i] is not null) yield return _adaBias[i]!;
        }
        if (_finalWeight is not null) yield return _finalWeight;
        if (_finalBias is not null) yield return _finalBias;
    }

    /// <summary>Refines the raw noisy pixel patches into the decoder's patch-space output.</summary>
    /// <param name="pixelPatches">Flattened noisy pixel patches [B, N, patch²·3] (Z-Image patchify order).</param>
    /// <param name="condTokens">Backbone output image tokens [B, N, backboneHidden].</param>
    /// <returns>Decoded patches [B, N, patch²·3] — unpatchify and negate to get the x0 prediction.</returns>
    public Tensor Forward(IBackend backend, Tensor pixelPatches, Tensor condTokens)
    {
        if (_condWeight is null || _embedWeight is null || _finalWeight is null)
            throw new InvalidOperationException("ZetaChromaDecoderHead weights not loaded.");

        int batch = (int)pixelPatches.Shape[0];
        int numPatches = (int)pixelPatches.Shape[1];
        int patchDim = (int)pixelPatches.Shape[2];
        int channels = _channels;
        int posFeatures = _maxFreqs * _maxFreqs;
        if (patchDim != _patchDim)
            throw new ArgumentException($"pixelPatches last dim {patchDim} != configured patch dim {_patchDim}.");
        if ((int)condTokens.Shape[1] != numPatches)
            throw new ArgumentException($"condTokens seq len {condTokens.Shape[1]} != pixel patch count {numPatches}.");

        // ── 1. Embed: [patch pixels, constant DCT features] → channels ──
        TensorShape embedInShape = new TensorShape(batch, numPatches, patchDim + posFeatures);
        Tensor embedInput = new Tensor(embedInShape, DType.F32);
        FillEmbedInput(embedInput, pixelPatches, batch, numPatches, patchDim, posFeatures);

        TensorShape hShape = new TensorShape(batch, numPatches, channels);
        Tensor h = new Tensor(hShape, DType.F32);
        backend.Linear(h, embedInput, _embedWeight, _embedBias);
        embedInput.Dispose();

        // ── 2. Per-patch conditioning from the backbone hidden state ──
        Tensor cond = new Tensor(hShape, DType.F32);
        backend.Linear(cond, condTokens, _condWeight, _condBias);
        Tensor condSilu = new Tensor(hShape, DType.F32);
        backend.Silu(condSilu, cond);
        cond.Dispose();

        // ── 3. AdaLN residual blocks ──
        TensorShape adaShape = new TensorShape(batch, numPatches, 3 * channels);
        Tensor ada = new Tensor(adaShape, DType.F32);
        Tensor normed = new Tensor(hShape, DType.F32);
        Tensor mlpMid = new Tensor(hShape, DType.F32);
        Tensor mlpAct = new Tensor(hShape, DType.F32);
        Tensor mlpOut = new Tensor(hShape, DType.F32);

        for (int i = 0; i < _resBlockCount; i++)
        {
            backend.Linear(ada, condSilu, _adaWeight[i]!, _adaBias[i]);

            DiTUtils.LayerNormNoAffine(normed, h, batch, numPatches, channels, LnEps);
            ApplyAffineAndModulation(normed, ada, _lnWeight[i]!, _lnBias[i], batch, numPatches, channels);

            backend.Linear(mlpMid, normed, _mlp0Weight[i]!, _mlp0Bias[i]);
            backend.Silu(mlpAct, mlpMid);
            backend.Linear(mlpOut, mlpAct, _mlp2Weight[i]!, _mlp2Bias[i]);

            ApplyGatedResidualInPlace(h, mlpOut, ada, batch, numPatches, channels);
        }

        ada.Dispose();
        mlpMid.Dispose();
        mlpAct.Dispose();
        mlpOut.Dispose();
        condSilu.Dispose();

        // ── 4. Final layer: LayerNorm-no-affine → Linear → patch space ──
        DiTUtils.LayerNormNoAffine(normed, h, batch, numPatches, channels, LnEps);
        h.Dispose();

        TensorShape outShape = new TensorShape(batch, numPatches, patchDim);
        Tensor output = new Tensor(outShape, DType.F32);
        backend.Linear(output, normed, _finalWeight, _finalBias);
        normed.Dispose();
        return output;
    }

    /// <summary>Constant positional features for the one-token-per-patch NerfEmbedder: with an internal patch size
    /// of 1, the shared cosine basis degenerates to <c>f[u·F+v] = 1/(1+u·v)</c> (all cosines are cos(0)=1).</summary>
    public static float[] BuildConstantDctFeatures(int maxFreqs)
    {
        float[] features = new float[maxFreqs * maxFreqs];
        for (int u = 0; u < maxFreqs; u++)
            for (int v = 0; v < maxFreqs; v++)
                features[u * maxFreqs + v] = 1.0f / (1.0f + u * v);
        return features;
    }

    private void FillEmbedInput(Tensor embedInput, Tensor pixelPatches, int batch, int numPatches,
        int patchDim, int posFeatures)
    {
        float[] dct = BuildConstantDctFeatures(_maxFreqs);
        float* src = (float*)pixelPatches.DataPointer;
        float* dst = (float*)embedInput.DataPointer;
        int rowIn = patchDim;
        int rowOut = patchDim + posFeatures;
        long rows = (long)batch * numPatches;
        for (long r = 0; r < rows; r++)
        {
            long srcBase = r * rowIn;
            long dstBase = r * rowOut;
            long copyBytes = (long)rowIn * sizeof(float);
            Buffer.MemoryCopy(src + srcBase, dst + dstBase, copyBytes, copyBytes);
            for (int f = 0; f < posFeatures; f++)
                dst[dstBase + rowIn + f] = dct[f];
        }
    }

    /// <summary>In place on <paramref name="normed"/>: applies the affine in_ln (weight/bias) followed by the
    /// AdaLN shift+scale modulation <c>x·w + b → ·(1+scale) + shift</c>. Shift = ada chunk 0, scale = chunk 1.</summary>
    private static void ApplyAffineAndModulation(Tensor normed, Tensor ada, Tensor lnWeight, Tensor? lnBias,
        int batch, int numPatches, int channels)
    {
        float* nPtr = (float*)normed.DataPointer;
        float* aPtr = (float*)ada.DataPointer;
        float* wPtr = (float*)lnWeight.DataPointer;
        float* bPtr = lnBias is not null ? (float*)lnBias.DataPointer : null;
        long rows = (long)batch * numPatches;
        for (long r = 0; r < rows; r++)
        {
            long nBase = r * channels;
            long aBase = r * 3L * channels;
            for (int d = 0; d < channels; d++)
            {
                float affine = nPtr[nBase + d] * wPtr[d] + (bPtr is not null ? bPtr[d] : 0f);
                float shift = aPtr[aBase + d];
                float scale = aPtr[aBase + channels + d];
                nPtr[nBase + d] = affine * (1.0f + scale) + shift;
            }
        }
    }

    /// <summary>In place on <paramref name="h"/>: <c>h += gate ⊙ mlpOut</c> with gate = ada chunk 2.</summary>
    private static void ApplyGatedResidualInPlace(Tensor h, Tensor mlpOut, Tensor ada,
        int batch, int numPatches, int channels)
    {
        float* hPtr = (float*)h.DataPointer;
        float* mPtr = (float*)mlpOut.DataPointer;
        float* aPtr = (float*)ada.DataPointer;
        long rows = (long)batch * numPatches;
        for (long r = 0; r < rows; r++)
        {
            long hBase = r * channels;
            long gateBase = r * 3L * channels + 2L * channels;
            for (int d = 0; d < channels; d++)
                hPtr[hBase + d] += aPtr[gateBase + d] * mPtr[hBase + d];
        }
    }

    /// <summary>Releases weight references.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _condWeight = null;
            _condBias = null;
            _embedWeight = null;
            _embedBias = null;
            for (int i = 0; i < _resBlockCount; i++)
            {
                _lnWeight[i] = null;
                _lnBias[i] = null;
                _mlp0Weight[i] = null;
                _mlp0Bias[i] = null;
                _mlp2Weight[i] = null;
                _mlp2Bias[i] = null;
                _adaWeight[i] = null;
                _adaBias[i] = null;
            }
            _finalWeight = null;
            _finalBias = null;
        }
        GC.SuppressFinalize(this);
    }
}
