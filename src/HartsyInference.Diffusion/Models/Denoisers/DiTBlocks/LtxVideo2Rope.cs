using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>LTX-2.3 audio/video rotary position embedding (<c>LTX2AudioVideoRotaryPosEmbed</c> +
/// <c>apply_interleaved_rotary_emb</c> / <c>apply_split_rotary_emb</c> in diffusers <c>transformer_ltx2.py</c>). The
/// per-token coordinates are computed in <b>pixel / time space</b> (latent patch <c>[start, end)</c> bounds × VAE
/// scale factors, temporal axis shifted by <c>causal_offset</c> and divided by fps for video or converted to seconds
/// for audio, then patch-midpoint and base-shape normalized) — identical for both apply flavors.
///
/// <para><b>Two apply flavors</b> (selected by <see cref="RopeType"/>, checkpoint-configured):
/// <list type="bullet">
/// <item><b>Interleaved</b> (base LTX / diffusers default): cos/sin are <c>[S, dim]</c>, per-axis values interleaved
/// (k-outer, axis-inner) and pair-duplicated with a leading <c>dim % (2·numAxes)</c> identity pad; rotation pairs
/// adjacent lanes <c>(2j, 2j+1)</c> across the full dim.</item>
/// <item><b>Split</b> (LTX-2.3 22B — <c>rope_type=split</c> in the checkpoint metadata): cos/sin are <c>[S, dim/2]</c>
/// (NO duplication), front-padded with <c>dim/2 − freqsPerAxis·numAxes</c> identity lanes, then laid out per head so
/// head <c>h</c> owns lanes <c>[h·headDim/2, (h+1)·headDim/2)</c>; rotation pairs <c>(h·headDim+i, h·headDim+i+headDim/2)</c>
/// — the two halves WITHIN each head — using compact <c>headDim/2</c> frequencies. Applying the wrong flavor scrambles
/// spatial positions (the LTX-2.3 32px-lattice bug).</item></list></para>
///
/// <para>Four flavors are needed by the DiT: video self (dim 4096, 3 axes, 32×128 heads), audio self (dim 2048, 1
/// axis, 32×64 heads), and the two cross-modal attentions (dim 2048, 1 <b>temporal-only</b> axis, 32×64 heads — the
/// cross-attn rope uses only <c>coords[:, 0]</c>). The audio cross rope is numerically identical to the audio self
/// rope. <see cref="ApplyRotary"/> runs on the full-dim <c>[S, dim]</c> Q/K before the head split (equivalent to the
/// reference's per-head apply).</para></summary>
public sealed unsafe class LtxVideo2Rope
{
    public enum Modality { Video, Audio }

    /// <summary>Rotary apply convention. <see cref="Split"/> is LTX-2.3 (22B checkpoint <c>rope_type=split</c>);
    /// <see cref="Interleaved"/> is base LTX / the diffusers default.</summary>
    public enum RopeType { Interleaved, Split }

    private readonly Modality _modality;
    private readonly RopeType _ropeType;

    /// <summary>Which apply flavor this rope uses. The fused QK-norm+rope+head-major backend op only serves
    /// <see cref="RopeType.Split"/>; an interleaved rope must keep the unfused sequence.</summary>
    public RopeType Flavor => _ropeType;
    private readonly bool _temporalOnly;    // video cross-attn: use only the temporal axis (1-axis rope)
    private readonly int _dim;              // full inner dim (Q/K width the rope is applied to)
    private readonly int _numHeads, _headDim;   // head layout (split mode pairs the two halves within each head)
    private readonly int _cosWidth;         // width of the cos/sin table: interleaved = dim, split = dim/2
    private readonly int _baseFrames, _baseHeight, _baseWidth;
    private readonly int _patchSize, _patchSizeT;
    private readonly int _causalOffset;
    private readonly int[] _scaleFactors;   // video: (t, h, w); audio: (t,)
    private readonly int _samplingRate, _hopLength;
    private readonly int _numAxes;          // 3 (video self), 1 (audio / video-cross)
    private readonly int _freqsPerAxis;     // dim / (2 · numAxes)
    private readonly int _pad;              // interleaved leading identity pad = dim % (2 · numAxes)
    private readonly int _padSplit;         // split leading identity pad = dim/2 − freqsPerAxis · numAxes
    private readonly double[] _freqBase;    // theta^linspace(0,1,n) · π/2

    /// <summary>Video self-attention RoPE: 3 axes (frame, height, width).</summary>
    public static LtxVideo2Rope ForVideoSelf(int dim, double theta, int baseFrames, int baseHeight, int baseWidth,
        int[] scaleFactors, int causalOffset, RopeType ropeType, int numHeads, int headDim,
        int patchSize = 1, int patchSizeT = 1) =>
        new(Modality.Video, ropeType, temporalOnly: false, dim, numHeads, headDim, theta, baseFrames, baseHeight,
            baseWidth, scaleFactors, causalOffset, samplingRate: 0, hopLength: 0, patchSize, patchSizeT);

    /// <summary>Video cross-attention RoPE: 1 temporal-only axis (uses <c>coords[:, 0]</c>). The head layout is the
    /// cross-attn's (a2v) query head config, not the video self-attn's.</summary>
    public static LtxVideo2Rope ForVideoCross(int dim, double theta, int baseFrames, int baseHeight, int baseWidth,
        int[] scaleFactors, int causalOffset, RopeType ropeType, int numHeads, int headDim,
        int patchSize = 1, int patchSizeT = 1) =>
        new(Modality.Video, ropeType, temporalOnly: true, dim, numHeads, headDim, theta, baseFrames, baseHeight,
            baseWidth, scaleFactors, causalOffset, samplingRate: 0, hopLength: 0, patchSize, patchSizeT);

    /// <summary>Audio RoPE (self and cross are identical): 1 axis, coordinates in seconds.</summary>
    public static LtxVideo2Rope ForAudio(int dim, double theta, int baseFrames, int audioScaleFactor,
        int causalOffset, int samplingRate, int hopLength, RopeType ropeType, int numHeads, int headDim,
        int patchSizeT = 1) =>
        new(Modality.Audio, ropeType, temporalOnly: false, dim, numHeads, headDim, theta, baseFrames, baseHeight: 0,
            baseWidth: 0, [audioScaleFactor], causalOffset, samplingRate, hopLength, patchSize: 1, patchSizeT);

    /// <summary>Text-connector 1D RoPE (<c>LTX2RotaryPosEmbed1d</c>): 1 axis, coordinate = <c>pos / base_seq_len</c>
    /// (the base sequence length is stored in the frame-base slot).</summary>
    public static LtxVideo2Rope ForConnector1d(int dim, double theta, int baseSeqLen, RopeType ropeType,
        int numHeads, int headDim) =>
        new(Modality.Audio, ropeType, temporalOnly: false, dim, numHeads, headDim, theta, baseFrames: baseSeqLen,
            baseHeight: 0, baseWidth: 0, [1], causalOffset: 0, samplingRate: 0, hopLength: 0, patchSize: 1, patchSizeT: 1);

    /// <summary>Builds connector cos/sin <c>[seqLen, cosWidth]</c> with positions <c>i / base_seq_len</c>.</summary>
    public (Tensor Cos, Tensor Sin) BuildConnector(int seqLen)
    {
        (Tensor cos, Tensor sin) = Alloc(seqLen);
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;
        for (int i = 0; i < seqLen; i++)
            WriteToken(cp, sp, i, [(double)i / _baseFrames]);
        return (cos, sin);
    }

    private LtxVideo2Rope(Modality modality, RopeType ropeType, bool temporalOnly, int dim, int numHeads, int headDim,
        double theta, int baseFrames, int baseHeight, int baseWidth, int[] scaleFactors, int causalOffset,
        int samplingRate, int hopLength, int patchSize, int patchSizeT)
    {
        _modality = modality;
        _ropeType = ropeType;
        _temporalOnly = temporalOnly;
        _dim = dim;
        _numHeads = numHeads;
        _headDim = headDim;
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
        _cosWidth = ropeType == RopeType.Split ? dim / 2 : dim;
        _padSplit = _cosWidth - _freqsPerAxis * _numAxes;      // front identity pad for the split (dim/2) layout

        if (ropeType == RopeType.Split)
        {
            if (headDim % 2 != 0)
                throw new ArgumentException($"Split RoPE requires an even head dim; got {headDim}.", nameof(headDim));
            if (numHeads * headDim != dim)
                throw new ArgumentException($"Split RoPE: numHeads·headDim ({numHeads}·{headDim}) != dim ({dim}).");
            if (_padSplit < 0)
                throw new ArgumentException($"Split RoPE: negative pad ({_padSplit}) for dim {dim}, axes {_numAxes}.");
        }

        _freqBase = new double[_freqsPerAxis];
        for (int k = 0; k < _freqsPerAxis; k++)
        {
            double exp = _freqsPerAxis == 1 ? 0.0 : (double)k / (_freqsPerAxis - 1);  // linspace(0,1,n)
            _freqBase[k] = Math.Pow(theta, exp) * Math.PI / 2.0;
        }
    }

    public int Dim => _dim;

    /// <summary>Builds video cos/sin <c>[F·H·W, cosWidth]</c> (row-major <c>(f,h,w)</c>). For the temporal-only (cross)
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

    /// <summary>Builds audio cos/sin <c>[numFrames, cosWidth]</c>; coordinates are patch-midpoint timestamps in seconds.</summary>
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

    /// <summary>Writes one token's cos/sin row. Interleaved: leading identity pad, then k-outer/axis-inner pairs,
    /// each pair-duplicated across two lanes (width <c>dim</c>). Split: leading identity pad, then k-outer/axis-inner
    /// singles, NO duplication (width <c>dim/2</c>) — the per-head halving happens in <see cref="ApplyRotary"/>.</summary>
    private void WriteToken(float* cp, float* sp, long token, double[] normCoord)
    {
        long baseOff = token * _cosWidth;
        if (_ropeType == RopeType.Interleaved)
        {
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
        else   // Split: cos/sin width dim/2, front-padded, no duplication.
        {
            for (int d = 0; d < _padSplit; d++) { cp[baseOff + d] = 1f; sp[baseOff + d] = 0f; }
            int outIdx = _padSplit;
            for (int k = 0; k < _freqsPerAxis; k++)
                for (int axis = 0; axis < _numAxes; axis++)
                {
                    double phase = _freqBase[k] * (normCoord[axis] * 2.0 - 1.0);
                    cp[baseOff + outIdx] = (float)Math.Cos(phase);
                    sp[baseOff + outIdx] = (float)Math.Sin(phase);
                    outIdx++;
                }
        }
    }

    private (Tensor, Tensor) Alloc(int seq)
    {
        Tensor cos = new(new TensorShape(seq, _cosWidth), DType.F32);
        Tensor sin = new(new TensorShape(seq, _cosWidth), DType.F32);
        return (cos, sin);
    }

    /// <summary>Applies the rotation in-place to <paramref name="x"/> <c>[S, dim]</c> (full-dim Q or K before the head
    /// split). Interleaved: cos/sin are <c>[S, dim]</c>, pairs are adjacent lanes. Split: cos/sin are <c>[S, dim/2]</c>,
    /// head <c>h</c> rotates its two halves <c>(h·headDim+i, h·headDim+i+headDim/2)</c> using cos/sin lane
    /// <c>h·(headDim/2)+i</c> (matches diffusers <c>apply_split_rotary_emb</c> after the per-head reshape).</summary>
    public void ApplyRotary(IBackend backend, Tensor x, Tensor cos, Tensor sin)
    {
        int seq = (int)x.Shape[0];
        if ((int)x.Shape[1] != _dim)
            throw new ArgumentException($"x dim {x.Shape[1]} != rope dim {_dim}.", nameof(x));
        // Device-resident RoPE. The host DataPointer loop D2H'd + then re-uploaded the [S,dim] Q/K on every attention;
        // on the block-swap-bound LTX-2.3 22B those re-uploads fought the 19 GB/forward weight stream on PCIe.
        //  - Interleaved: adjacent-pair (2j,2j+1) over the full dim, duplicated-pair cos[2j]==cos[2j+1] → the shared
        //    wan_rope_interleaved kernel (headDim=dim, heads=1), identical to LtxRope / LTX-0.9.
        //  - Split: rotate-half within each head, per-head cos[S,dim/2] with one angle per pair → the dedicated
        //    ltx2_split_rope kernel.
        if (_ropeType == RopeType.Interleaved)
            backend.WanRopeInterleaved(x, cos, sin, seq, 1, _dim);
        else
            backend.Ltx2SplitRope(x, cos, sin, seq, _numHeads, _headDim);
    }
}
