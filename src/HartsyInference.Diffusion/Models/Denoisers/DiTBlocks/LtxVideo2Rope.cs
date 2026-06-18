using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>LTX-2.3 audio/video rotary position embedding (<c>LTX2AudioVideoRotaryPosEmbed</c> +
/// <c>apply_interleaved_rotary_emb</c> in diffusers <c>transformer_ltx2.py</c>). Same interleaved-pair complex form
/// and full-dim apply as <see cref="LtxRope"/> (LTX-1), but the per-token coordinates are computed in <b>pixel /
/// time space</b>: each latent patch's <c>[start, end)</c> bounds are scaled by the VAE factors, the temporal axis
/// is shifted by <c>causal_offset</c> and divided by fps (video) or converted to seconds (audio), the patch
/// midpoint is taken, then normalized by the base data shape. Used for video self-attn (dim 4096, 3 axes), audio
/// self-attn (dim 2048, 1 axis), and the two cross-modal attentions (dim 2048 each, video- and audio-coordinated).</summary>
///
/// <remarks>Per axis there are <c>dim / (2·numAxes)</c> log-spaced frequencies (<c>θ^linspace(0,1,n)·π/2</c>); the
/// phase is <c>freq·(normCoord·2−1)</c>. cos/sin are <c>[S, dim]</c> with the per-axis values interleaved
/// (k-outer, axis-inner) and pair-duplicated; a leading <c>dim % (2·numAxes)</c> identity pad (cos=1, sin=0) fills
/// the remainder. <see cref="ApplyRotary"/> matches <see cref="LtxRope.ApplyRotary"/> exactly.</remarks>
public sealed unsafe class LtxVideo2Rope
{
    public enum Modality { Video, Audio }

    private readonly Modality _modality;
    private readonly int _dim;
    private readonly double _theta;
    private readonly int _baseFrames, _baseHeight, _baseWidth;
    private readonly int _patchSize, _patchSizeT;
    private readonly int _causalOffset;
    private readonly int[] _scaleFactors;   // video: (t, h, w); audio: (t,)
    private readonly int _samplingRate, _hopLength;
    private readonly int _numAxes;          // 3 video, 1 audio
    private readonly int _freqsPerAxis;     // dim / (2 · numAxes)
    private readonly int _pad;              // dim % (2 · numAxes)
    private readonly double[] _freqBase;    // length freqsPerAxis: theta^linspace(0,1,n) · π/2

    /// <summary>Video constructor: 3-axis (frame, height, width) RoPE.</summary>
    public static LtxVideo2Rope ForVideo(int dim, double theta, int baseFrames, int baseHeight, int baseWidth,
        int[] scaleFactors, int causalOffset, int patchSize = 1, int patchSizeT = 1) =>
        new(Modality.Video, dim, theta, baseFrames, baseHeight, baseWidth, scaleFactors, causalOffset,
            samplingRate: 0, hopLength: 0, patchSize, patchSizeT);

    /// <summary>Audio constructor: 1-axis (time) RoPE; coordinates are in seconds.</summary>
    public static LtxVideo2Rope ForAudio(int dim, double theta, int baseFrames, int audioScaleFactor,
        int causalOffset, int samplingRate, int hopLength, int patchSizeT = 1) =>
        new(Modality.Audio, dim, theta, baseFrames, baseHeight: 0, baseWidth: 0, [audioScaleFactor], causalOffset,
            samplingRate, hopLength, patchSize: 1, patchSizeT);

    private LtxVideo2Rope(Modality modality, int dim, double theta, int baseFrames, int baseHeight, int baseWidth,
        int[] scaleFactors, int causalOffset, int samplingRate, int hopLength, int patchSize, int patchSizeT)
    {
        _modality = modality;
        _dim = dim;
        _theta = theta;
        _baseFrames = baseFrames;
        _baseHeight = baseHeight;
        _baseWidth = baseWidth;
        _scaleFactors = scaleFactors;
        _causalOffset = causalOffset;
        _samplingRate = samplingRate;
        _hopLength = hopLength;
        _patchSize = patchSize;
        _patchSizeT = patchSizeT;
        _numAxes = modality == Modality.Video ? 3 : 1;
        int numRopeElems = _numAxes * 2;
        _freqsPerAxis = dim / numRopeElems;
        _pad = dim % numRopeElems;

        _freqBase = new double[_freqsPerAxis];
        for (int k = 0; k < _freqsPerAxis; k++)
        {
            double exp = _freqsPerAxis == 1 ? 0.0 : (double)k / (_freqsPerAxis - 1);  // linspace(0,1,n)
            _freqBase[k] = Math.Pow(_theta, exp) * Math.PI / 2.0;
        }
    }

    public int Dim => _dim;

    /// <summary>Builds video cos/sin <c>[F·H·W, dim]</c> for a latent grid (row-major <c>(f,h,w)</c>).</summary>
    public (Tensor Cos, Tensor Sin) BuildVideo(int numFrames, int height, int width, double fps)
    {
        if (_modality != Modality.Video) throw new InvalidOperationException("BuildVideo on an audio RoPE.");
        int seq = numFrames * height * width;
        (Tensor cos, Tensor sin) = Alloc(seq);
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;

        int sfT = _scaleFactors[0], sfH = _scaleFactors[1], sfW = _scaleFactors[2];
        for (int fi = 0; fi < numFrames; fi++)
        {
            // Temporal: pixel bounds, causal shift+clamp, divide by fps; then midpoint and base-normalize.
            double tStart = Math.Max(0.0, (double)fi * _patchSizeT * sfT + _causalOffset - sfT) / fps;
            double tEnd = Math.Max(0.0, (double)(fi + 1) * _patchSizeT * sfT + _causalOffset - sfT) / fps;
            double normT = ((tStart + tEnd) / 2.0) / _baseFrames;
            for (int hi = 0; hi < height; hi++)
            {
                double normH = (((double)hi + 0.5) * _patchSize * sfH) / _baseHeight;
                for (int wi = 0; wi < width; wi++)
                {
                    double normW = (((double)wi + 0.5) * _patchSize * sfW) / _baseWidth;
                    long token = ((long)fi * height + hi) * width + wi;
                    WriteToken(cp, sp, token, [normT, normH, normW]);
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
            double startS = startMel * melToSec, endS = endMel * melToSec;
            double norm = ((startS + endS) / 2.0) / _baseFrames;
            WriteToken(cp, sp, fi, [norm]);
        }
        return (cos, sin);
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
