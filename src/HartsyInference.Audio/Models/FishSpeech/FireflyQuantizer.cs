using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.FishSpeech;

/// <summary>Firefly-gan-vq <c>DownsampleFiniteScalarQuantize</c> DECODE path (Fish-Speech 1.5, fish-speech
/// <c>fsq.py</c>). The codec is a <b>grouped</b> FSQ: the 512-dim latent is split into <c>n_groups=8</c> chunks of
/// 64, and each chunk has its own <c>ResidualFSQ</c> (here <c>n_codebooks=1</c>, so a single FSQ stage with
/// levels <c>[8,5,5,5]</c> → 1000-entry vocab). Decode:
/// <code>
///   indices [b, g, l]  (g = 8 groups)
///   per group: digit_k = (idx // basis_k) % levels_k;  code_k = (digit_k - levels_k//2) / (levels_k//2)
///              latent_g = project_out_g(code)              # Linear 4 -> 64
///   z = concat_g(latent_g)                                # [b, l, 512] channels-last -> [b, 512, l]
///   z = upsample(z)                                       # 2x (FishTransConvNet k2 s2 + ConvNeXtBlock) -> [b,512,4l]
/// </code>
///
/// <para>FSQ index→code uses the lucidrains <c>_scale_and_shift_inverse</c> convention (<c>half_width = L//2</c>),
/// which differs from the bounded-tanh <see cref="Codecs.Fsq"/> dequant for even <c>L</c>; it is computed inline
/// here to match the real <c>vector_quantize_pytorch.GroupedResidualFSQ.get_output_from_indices</c>.</para></summary>
public sealed unsafe class FireflyQuantizer : IDisposable
{
    private readonly int[] _levels;
    private readonly int _codebookDim;       // len(levels) = 4
    private readonly int _nGroups;           // 8
    private readonly int _groupDim;          // 64 = inputDim / nGroups
    private readonly int _inputDim;          // 512
    private readonly int[] _upsampleFactors; // [2, 2]

    private readonly Tensor?[] _projOutW;    // per-group residual-FSQ out projection [groupDim, codebookDim]
    private readonly Tensor?[] _projOutB;
    private readonly Tensor?[] _upConvW;     // per-stage FishTransConvNet weight [in, out, k]
    private readonly Tensor?[] _upConvB;
    private readonly FireflyConvNeXtBlock[] _upBlocks;
    private int _disposed;

    public FireflyQuantizer(int[]? levels = null, int nGroups = 8, int[]? upsampleFactors = null, int inputDim = 512)
    {
        _levels = levels ?? [8, 5, 5, 5];
        _codebookDim = _levels.Length;
        _nGroups = nGroups;
        _inputDim = inputDim;
        if (inputDim % nGroups != 0) throw new ArgumentException($"inputDim {inputDim} must be divisible by nGroups {nGroups}.");
        _groupDim = inputDim / nGroups;
        _upsampleFactors = upsampleFactors ?? [2, 2];
        _projOutW = new Tensor?[_nGroups];
        _projOutB = new Tensor?[_nGroups];
        _upConvW = new Tensor?[_upsampleFactors.Length];
        _upConvB = new Tensor?[_upsampleFactors.Length];
        _upBlocks = new FireflyConvNeXtBlock[_upsampleFactors.Length];
        for (int i = 0; i < _upsampleFactors.Length; i++) _upBlocks[i] = new FireflyConvNeXtBlock(_inputDim);
    }

    /// <summary>Total temporal upsample factor (product of the ConvNeXt upsample strides). 4 for firefly.</summary>
    public int UpsampleFactor { get { int p = 1; foreach (int f in _upsampleFactors) p *= f; return p; } }

    /// <summary>Optional per-stage activation hook for parity debugging (key = stage name). Not used in production.</summary>
    public Action<string, Tensor>? DebugHook { get; set; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "quantizer")
    {
        for (int g = 0; g < _nGroups; g++)
        {
            string po = $"{prefix}.residual_fsq.rvqs.{g}.project_out";
            _projOutW[g] = WhisperOps.EnsureF32(w[$"{po}.weight"]);
            _projOutB[g] = w.TryGetValue($"{po}.bias", out Tensor? b) ? WhisperOps.EnsureF32(b) : null;
        }
        // upsample.{i} = Sequential(FishTransConvNet at .0, ConvNeXtBlock at .1).
        for (int i = 0; i < _upsampleFactors.Length; i++)
        {
            string up = $"{prefix}.upsample.{i}";
            _upConvW[i] = FireflyOps.LoadTransConvWeight(w, $"{up}.0.conv");
            _upConvB[i] = WhisperOps.EnsureF32(w[$"{up}.0.conv.bias"]);
            _upBlocks[i].LoadWeights(w, $"{up}.1");
        }
    }

    /// <summary>Dequantizes <c>[nGroups, T]</c> indices and runs the ConvNeXt upsample stack, returning the firefly
    /// HiFi-GAN decoder input <c>[1, inputDim, T * UpsampleFactor]</c>.</summary>
    public Tensor DequantToDecoderInput(IBackend backend, int[,] codes, int t)
    {
        if (codes.GetLength(0) != _nGroups)
            throw new ArgumentException($"codes must have {_nGroups} rows, got {codes.GetLength(0)}.");
        if (t <= 0) throw new ArgumentException("t must be positive.", nameof(t));

        // Per-axis place values and half-widths. fish-speech's GroupedResidualFSQ(levels=[8,5,5,5]) leaves
        // preserve_symmetry at its default False, so index k maps to digit_k = (idx // basis_k) % levels_k then
        // code_k = (digit_k - levels_k//2) / (levels_k//2) — the half_width convention (NOT the evenly-spaced
        // 2/(L-1) mapping, which only agrees on odd levels and mis-scales the L=8 axis).
        Span<int> basis = stackalloc int[_codebookDim];
        Span<int> halfWidth = stackalloc int[_codebookDim];
        basis[0] = 1;
        for (int k = 1; k < _codebookDim; k++) basis[k] = basis[k - 1] * _levels[k - 1];
        for (int k = 0; k < _codebookDim; k++) halfWidth[k] = _levels[k] / 2;

        // Build the channels-first latent [1, inputDim, T]; group g occupies channels [g*groupDim, (g+1)*groupDim).
        Tensor latent = new(new TensorShape(1, _inputDim, t), DType.F32);
        float* lp = (float*)latent.DataPointer;

        Tensor code = new(new TensorShape(1, t, _codebookDim), DType.F32);
        float* cdp = (float*)code.DataPointer;
        for (int g = 0; g < _nGroups; g++)
        {
            for (int l = 0; l < t; l++)
            {
                int idx = codes[g, l];
                for (int k = 0; k < _codebookDim; k++)
                {
                    int digit = (idx / basis[k]) % _levels[k];
                    cdp[(long)l * _codebookDim + k] = (digit - halfWidth[k]) / (float)halfWidth[k];
                }
            }
            // project_out: Linear codebookDim -> groupDim.
            Tensor proj = WhisperOps.ProjectLinear(backend, code, _projOutW[g]!, _projOutB[g], 1, t, _codebookDim, _groupDim);
            float* pp = (float*)proj.DataPointer;
            for (int j = 0; j < _groupDim; j++)
                for (int l = 0; l < t; l++)
                    lp[(long)(g * _groupDim + j) * t + l] = pp[(long)l * _groupDim + j];
            proj.Dispose();
        }
        code.Dispose();
        DebugHook?.Invoke("fsq_out", latent);

        // Upsample stack: FishTransConvNet(k=2,s=2) then ConvNeXtBlock, x2 -> [1, inputDim, T*4].
        Tensor x = latent;
        int curT = t;
        for (int i = 0; i < _upsampleFactors.Length; i++)
        {
            int stride = _upsampleFactors[i];
            int kernel = (int)_upConvW[i]!.Shape[2];
            Tensor up = FireflyOps.TransConv(backend, x, _upConvW[i]!, _upConvB[i], _inputDim, curT, _inputDim, kernel, stride);
            x.Dispose();
            curT = (int)up.Shape[2];
            x = _upBlocks[i].Forward(backend, up, curT);
            up.Dispose();
            DebugHook?.Invoke($"q_up_{i}", x);
        }
        DebugHook?.Invoke("z_q", x);
        return x;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in _projOutW) if (t is not null) yield return t;
        foreach (Tensor? t in _projOutB) if (t is not null) yield return t;
        foreach (Tensor? t in _upConvW) if (t is not null) yield return t;
        foreach (Tensor? t in _upConvB) if (t is not null) yield return t;
        foreach (FireflyConvNeXtBlock blk in _upBlocks)
            foreach (Tensor t in blk.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (FireflyConvNeXtBlock blk in _upBlocks) blk.Dispose();
        GC.SuppressFinalize(this);
    }
}
