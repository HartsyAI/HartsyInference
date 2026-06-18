using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>LTX-2.3 audio/video rotary position embedding (<c>LTX2AudioVideoRotaryPosEmbed</c> +
/// <c>apply_interleaved_rotary_emb</c> in diffusers <c>transformer_ltx2.py</c>). Same interleaved-pair complex form
/// and full-dim apply as <see cref="LtxRope"/> (LTX-1); the per-token coordinates are computed in <b>pixel / time
/// space</b> (latent patch <c>[start, end)</c> bounds × VAE scale factors, temporal axis shifted by
/// <c>causal_offset</c> and divided by fps for video or converted to seconds for audio, then patch-midpoint and
/// base-shape normalized). Four flavors are needed by the DiT: video self (dim 4096, 3 axes), audio self (dim 2048,
/// 1 axis), and the two cross-modal attentions (dim 2048, 1 <b>temporal-only</b> axis — the cross-attn rope uses
/// only <c>coords[:, 0]</c>). The audio cross rope is numerically identical to the audio self rope.</summary>
///
/// <remarks>Per axis there are <c>dim / (2·numAxes)</c> log-spaced frequencies (<c>θ^linspace(0,1,n)·π/2</c>);
/// phase = <c>freq·(normCoord·2−1)</c>. cos/sin are <c>[S, dim]</c> with per-axis values interleaved (k-outer,
/// axis-inner) and pair-duplicated, a leading <c>dim % (2·numAxes)</c> identity pad (cos=1, sin=0) filling the
/// remainder. <see cref="ApplyRotary"/> matches <see cref="LtxRope.ApplyRotary"/> exactly.</remarks>
public sealed unsafe class LtxVideo2Rope
{
    public enum Modality { Video, Audio }

    private readonly Modality _modality;
    private readonly bool _temporalOnly;    // video cross-attn: use only the temporal axis (1-axis rope)
    private readonly int _dim;
    private readonly int _baseFrames, _baseHeight, _baseWidth;
    private readonly int _patchSize, _patchSizeT;
    private readonly int _causalOffset;
    private readonly int[] _scaleFactors;   // video: (t, h, w); audio: (t,)
    private readonly int _samplingRate, _hopLength;
    private readonly int _numAxes;          // 3 (video self), 1 (audio / video-cross)
    private readonly int _freqsPerAxis;     // dim / (2 · numAxes)
    private readonly int _pad;              // dim % (2 · numAxes)
    private readonly double[] _freqBase;    // theta^linspace(0,1,n) · π/2

    /// <summary>Video self-attention RoPE: 3 axes (frame, height, width).</summary>
    public static LtxVideo2Rope ForVideoSelf(int dim, double theta, int baseFrames, int baseHeight, int baseWidth,
        int[] scaleFactors, int causalOffset, int patchSize = 1, int patchSizeT = 1) =>
        new(Modality.Video, temporalOnly: false, dim, theta, baseFrames, baseHeight, baseWidth, scaleFactors,
            causalOffset, samplingRate: 0, hopLength: 0, patchSize, patchSizeT);

    /// <summary>Video cross-attention RoPE: 1 temporal-only axis (uses <c>coords[:, 0]</c>).</summary>
    public static LtxVideo2Rope ForVideoCross(int dim, double theta, int baseFrames, int baseHeight, int baseWidth,
        int[] scaleFactors, int causalOffset, int patchSize = 1, int patchSizeT = 1) =>
        new(Modality.Video, temporalOnly: true, dim, theta, baseFrames, baseHeight, baseWidth, scaleFactors,
            causalOffset, samplingRate: 0, hopLength: 0, patchSize, patchSizeT);

    /// <summary>Audio RoPE (self and cross are identical): 1 axis, coordinates in seconds.</summary>
    public static LtxVideo2Rope ForAudio(int dim, double theta, int baseFrames, int audioScaleFactor,
        int causalOffset, int samplingRate, int hopLength, int patchSizeT = 1) =>
        new(Modality.Audio, temporalOnly: false, dim, theta, baseFrames, baseHeight: 0, baseWidth: 0,
            [audioScaleFactor], causalOffset, samplingRate, hopLength, patchSize: 1, patchSizeT);

    /// <summary>Text-connector 1D RoPE (<c>LTX2RotaryPosEmbed1d</c>): 1 axis, coordinate = <c>pos / base_seq_len</c>
    /// (the base sequence length is stored in the frame-base slot).</summary>
    public static LtxVideo2Rope ForConnector1d(int dim, double theta, int baseSeqLen) =>
        new(Modality.Audio, temporalOnly: false, dim, theta, baseFrames: baseSeqLen, baseHeight: 0, baseWidth: 0,
            [1], causalOffset: 0, samplingRate: 0, hopLength: 0, patchSize: 1, patchSizeT: 1);

    /// <summary>Builds connector cos/sin <c>[seqLen, dim]</c> with positions <c>i / base_seq_len</c>.</summary>
    public (Tensor Cos, Tensor Sin) BuildConnector(int seqLen)
    {
        (Tensor cos, Tensor sin) = Alloc(seqLen);
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;
        for (int i = 0; i < seqLen; i++)
            WriteToken(cp, sp, i, [(double)i / _baseFrames]);
        return (cos, sin);
    }

    private LtxVideo2Rope(Modality modality, bool temporalOnly, int dim, double theta, int baseFrames, int baseHeight,
        int baseWidth, int[] scaleFactors, int causalOffset, int samplingRate, int hopLength, int patchSize, int patchSizeT)
    {
        _modality = modality;
        _temporalOnly = temporalOnly;
        _dim = dim;
        _baseFrames = baseFrames;
        _baseHeight = baseHeight;
        _baseWidth = baseWidth;
        _scaleFactors = scaleFactors;
        _causalOffset = causalOffset;
        _samplingRate = samplingRate;
        _hopLength = hopLength;
        _patchSize = patchSize;
        _patchSizeT = patchSizeT;
        _numAxes = (modality == Modality.Video && !temporalOnly) ? 3 : 1;
        int numRopeElems = _numAxes * 2;
        _freqsPerAxis = dim / numRopeElems;
        _pad = dim % numRopeElems;

        _freqBase = new double[_freqsPerAxis];
        for (int k = 0; k < _freqsPerAxis; k++)
        {
            double exp = _freqsPerAxis == 1 ? 0.0 : (double)k / (_freqsPerAxis - 1);  // linspace(0,1,n)
            _freqBase[k] = Math.Pow(theta, exp) * Math.PI / 2.0;
        }
    }

    public int Dim => _dim;

    /// <summary>Builds video cos/sin <c>[F·H·W, dim]</c> (row-major <c>(f,h,w)</c>). For the temporal-only (cross)
    /// flavor each token's single coordinate is its frame midpoint; otherwise all three axes are written.</summary>
    public (Tensor Cos, Tensor Sin) BuildVideo(int numFrames, int height, int width, double fps)
    {
        if (_modality != Modality.Video) throw new InvalidOperationException("BuildVideo on an audio RoPE.");
        int seq = numFrames * height * width;
        (Tensor cos, Tensor sin) = Alloc(seq);
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;

        int sfH = _scaleFactors[1], sfW = _scaleFactors[2];
        for (int fi = 0; fi < numFrames; fi++)
        {
            double normT = VideoTemporalNorm(fi, fps);
            for (int hi = 0; hi < height; hi++)
            {
                double normH = (((double)hi + 0.5) * _patchSize * sfH) / _baseHeight;
                for (int wi = 0; wi < width; wi++)
                {
                    double normW = (((double)wi + 0.5) * _patchSize * sfW) / _baseWidth;
                    long token = ((long)fi * height + hi) * width + wi;
                    if (_temporalOnly) WriteToken(cp, sp, token, [normT]);
                    else WriteToken(cp, sp, token, [normT, normH, normW]);
                }
            }
        }
        return (cos, sin);
    }

    /// <summary>Builds audio cos/sin <c>[numFrames, dim]</c>; coordinates are patch-midpoint timestamps in seconds.</summary>
    public (Tensor Cos, Tensor Sin) BuildAudio(int numFrames, int shift = 0)
    {
        if (_modality != Modality.Audio) throw new InvalidOperationException("BuildAudio on a video RoPE.");
        (Tensor cos, Tensor sin) = Alloc(numFrames);
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;

        int sf = _scaleFactors[0];
        double melToSec = (double)_hopLength / _samplingRate;
        for (int fi = 0; fi < numFrames; fi++)
        {
            int f = fi + shift;
            double startMel = Math.Max(0.0, (double)f * sf + _causalOffset - sf);
            double endMel = Math.Max(0.0, (double)(f + _patchSizeT) * sf + _causalOffset - sf);
            double norm = ((startMel * melToSec + endMel * melToSec) / 2.0) / _baseFrames;
            WriteToken(cp, sp, fi, [norm]);
        }
        return (cos, sin);
    }

    /// <summary>Video temporal patch-midpoint coordinate (pixel bounds, causal shift+clamp, ÷ fps), base-normalized.</summary>
    private double VideoTemporalNorm(int fi, double fps)
    {
        int sfT = _scaleFactors[0];
        double tStart = Math.Max(0.0, (double)fi * _patchSizeT * sfT + _causalOffset - sfT) / fps;
        double tEnd = Math.Max(0.0, (double)(fi + 1) * _patchSizeT * sfT + _causalOffset - sfT) / fps;
        return ((tStart + tEnd) / 2.0) / _baseFrames;
    }

    /// <summary>Writes one token's cos/sin row: leading identity pad, then k-outer/axis-inner interleaved pairs.</summary>
    private void WriteToken(float* cp, float* sp, long token, double[] normCoord)
    {
        long baseOff = token * _dim;
        for (int d = 0; d < _pad; d++) { cp[baseOff + d] = 1f; sp[baseOff + d] = 0f; }
        int outIdx = _pad;
        for (int k = 0; k < _freqsPerAxis; k++)
            for (int axis = 0; axis < _numAxes; axis++)
            {
                double phase = _freqBase[k] * (normCoord[axis] * 2.0 - 1.0);
                float c = (float)Math.Cos(phase);
                float s = (float)Math.Sin(phase);
                cp[baseOff + outIdx] = c; sp[baseOff + outIdx] = s;
                cp[baseOff + outIdx + 1] = c; sp[baseOff + outIdx + 1] = s;
                outIdx += 2;
            }
    }

    private (Tensor, Tensor) Alloc(int seq)
    {
        Tensor cos = new(new TensorShape(seq, _dim), DType.F32);
        Tensor sin = new(new TensorShape(seq, _dim), DType.F32);
        return (cos, sin);
    }

    /// <summary>Applies the interleaved-pair rotation in-place to <paramref name="x"/> <c>[S, dim]</c> (full-dim Q or
    /// K before head split). cos/sin are <c>[S, dim]</c>. Identical to <see cref="LtxRope.ApplyRotary"/>.</summary>
    public void ApplyRotary(Tensor x, Tensor cos, Tensor sin)
    {
        int seq = (int)x.Shape[0];
        if ((int)x.Shape[1] != _dim)
            throw new ArgumentException($"x dim {x.Shape[1]} != rope dim {_dim}.", nameof(x));
        float* xp = (float*)x.DataPointer;
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;
        int pairs = _dim / 2;

        for (int s = 0; s < seq; s++)
        {
            long off = (long)s * _dim;
            for (int j = 0; j < pairs; j++)
            {
                int i0 = (int)off + 2 * j;
                int i1 = i0 + 1;
                float re = xp[i0], im = xp[i1];
                float c = cp[i0], sn = sp[i0];   // cos[2j] == cos[2j+1], sin likewise
                xp[i0] = re * c - im * sn;
                xp[i1] = im * c + re * sn;
            }
        }
    }
}
