using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.CosyVoice;

/// <summary>S3Tokenizer (CosyVoice 2 variant) — converts a reference clip's mel into the discrete 25 Hz
/// FSQ speech-token stream the LM/flow consume. A supervised-ASR-style encoder (Whisper-class, 6 lower
/// Transformer blocks with RoPE) produces semantically-aligned features that an 8-channel Finite Scalar
/// Quantizer maps to <c>3^8 = 6561</c> codes. Only the encode direction is needed at inference.
///
/// <para><b>Exact + tested:</b> <see cref="PackFsqTokens"/> implements the FSQ formula
/// (<c>tanh → round → shift → base-3 pack</c>) verbatim from the architecture doc. <b>Scaffold:</b> the
/// encoder here is a 4× conv subsample + <c>proj_down</c>; the full 6-block RoPE transformer encoder is
/// checkpoint-gated (<c>speech_tokenizer_v2.onnx</c> is also available as an ONNX fallback).</para></summary>
public sealed unsafe class S3Tokenizer : IDisposable
{
    private const int FsqDim = 8;
    private const int FsqLevels = 3;
    private int _disposed;

    private Tensor?[] _subsampleW = new Tensor[2];
    private Tensor?[] _subsampleB = new Tensor[2];
    private Tensor? _projDownW, _projDownB;     // hidden → FsqDim

    public int VocabSize { get; } = 1;

    public S3Tokenizer()
    {
        VocabSize = 1;
        for (int i = 0; i < FsqDim; i++) VocabSize *= FsqLevels;     // 6561
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        for (int i = 0; i < 2; i++)
        {
            _subsampleW[i] = WhisperOps.EnsureF32(w[$"{p}subsample.{i}.weight"]);
            _subsampleB[i] = WhisperOps.EnsureF32(w[$"{p}subsample.{i}.bias"]);
        }
        _projDownW = WhisperOps.EnsureF32(w[$"{p}quantizer.project_down.weight"]);
        _projDownB = w.TryGetValue($"{p}quantizer.project_down.bias", out Tensor? b) ? WhisperOps.EnsureF32(b) : null;
    }

    /// <summary>Encodes a 100 Hz mel <c>[1, 80, T]</c> into 25 Hz speech-token IDs.</summary>
    public int[] Forward(IBackend backend, Tensor mel)
    {
        if (_projDownW is null) throw new InvalidOperationException("S3Tokenizer weights not loaded.");
        // 4× conv subsample (two stride-2 Conv1d) → ~25 Hz.
        Tensor x = mel;
        bool owns = false;
        for (int i = 0; i < 2; i++)
        {
            Tensor wgt = _subsampleW![i]!;
            int outCh = (int)wgt.Shape[0];
            int k = (int)wgt.Shape[2];
            int inLen = (int)x.Shape[2];
            int pad = (k - 1) / 2;
            int outLen = (inLen + 2 * pad - (k - 1) - 1) / 2 + 1;
            Tensor nx = new(new TensorShape(1, outCh, outLen), DType.F32);
            backend.Conv1d(nx, x, wgt, _subsampleB![i], stride: 2, padLeft: pad, padRight: pad, dilation: 1, groups: 1);
            if (owns) x.Dispose();
            backend.LeakyRelu(nx, nx, 0f);
            x = nx;
            owns = true;
        }

        // proj_down to the 8 FSQ channels, then pack each frame.
        int hidden = (int)x.Shape[1];
        int t = (int)x.Shape[2];
        Tensor xt = TransposeToSeq(x, hidden, t);          // [1, T, hidden]
        if (owns) x.Dispose();
        Tensor z = WhisperOps.ProjectLinear(backend, xt, _projDownW!, _projDownB, 1, t, hidden, FsqDim);
        xt.Dispose();
        int[] tokens = PackFsqTokens(z, FsqDim, FsqLevels);
        z.Dispose();
        return tokens;
    }

    /// <summary>FSQ tokenizer: for each frame, <c>token = Σ_j (round((L-1)/2·tanh(z_j)) + (L-1)/2)·L^j</c>.
    /// <paramref name="z"/> is the projected latent <c>[1, T, D]</c>; returns one base-L code per frame.</summary>
    public static int[] PackFsqTokens(Tensor z, int dim, int levels)
    {
        int t = (int)z.Shape[1];
        if ((int)z.Shape[2] != dim) throw new ArgumentException($"FSQ latent must have {dim} channels, got {z.Shape[2]}.");
        float half = (levels - 1) / 2f;
        float* zp = (float*)z.DataPointer;
        int[] tokens = new int[t];
        for (int frame = 0; frame < t; frame++)
        {
            int token = 0;
            int pow = 1;
            for (int j = 0; j < dim; j++)
            {
                float bounded = half * MathF.Tanh(zp[(long)frame * dim + j]);
                int zint = (int)MathF.Round(bounded);                       // {-(L-1)/2 .. +(L-1)/2}
                int shift = zint + (int)half;                               // {0 .. L-1}
                if (shift < 0) shift = 0; else if (shift > levels - 1) shift = levels - 1;
                token += shift * pow;
                pow *= levels;
            }
            tokens[frame] = token;
        }
        return tokens;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int i = 0; i < 2; i++)
        {
            if (_subsampleW[i] is not null) yield return _subsampleW[i]!;
            if (_subsampleB[i] is not null) yield return _subsampleB[i]!;
        }
        if (_projDownW is not null) yield return _projDownW;
        if (_projDownB is not null) yield return _projDownB;
    }

    private static Tensor TransposeToSeq(Tensor chFirst, int c, int t)
    {
        Tensor outT = new(new TensorShape(1, t, c), DType.F32);
        float* ip = (float*)chFirst.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int cc = 0; cc < c; cc++)
            for (int j = 0; j < t; j++)
                op[(long)j * c + cc] = ip[(long)cc * t + j];
        return outT;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
