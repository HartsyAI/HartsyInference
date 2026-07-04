using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>LTX-Video 3D rotary position embedding (<c>LTXVideoRotaryPosEmbed</c> + <c>apply_rotary_emb</c> in diffusers <c>transformer_ltx.py</c>). Distinct from <see cref="Multimodal3DRope"/>: LTX uses an **interleaved-pair complex** form (Flux-style) over the FULL inner dim (applied to Q/K before the head split), with a 3-axis grid <c>(f, h, w)</c> where each axis gets <c>dim//6</c> log-spaced frequencies and per-token phase <c>freq·(coord·2−1)</c>.
///
/// <para>cos/sin are <c>[S, dim]</c>: the <c>dim//6</c> per-axis frequencies are interleaved by axis, duplicated to pairs (<c>repeat_interleave(2)</c>); when <c>dim % 6 ≠ 0</c> the leading <c>dim%6</c> channels are an identity pad (cos=1, sin=0). Coordinates are normalized by the base frame/height/width and an optional per-axis interpolation scale.</para></summary>
public sealed unsafe class LtxRope
{
    private readonly int _dim;
    private readonly double _theta;
    private readonly int _baseFrames, _baseHeight, _baseWidth;
    private readonly int _patchSize, _patchSizeT;
    private readonly int _freqsPerAxis;     // dim // 6
    private readonly int _pad;              // dim % 6 (identity-pad channels at the front)
    private readonly double[] _freqBase;    // length freqsPerAxis: theta^linspace(0,1,n) · π/2

    public LtxRope(int dim, double theta, int baseFrames, int baseHeight, int baseWidth, int patchSize = 1, int patchSizeT = 1)
    {
        _dim = dim;
        _theta = theta;
        _baseFrames = baseFrames;
        _baseHeight = baseHeight;
        _baseWidth = baseWidth;
        _patchSize = patchSize;
        _patchSizeT = patchSizeT;
        _freqsPerAxis = dim / 6;
        _pad = dim % 6;

        _freqBase = new double[_freqsPerAxis];
        for (int k = 0; k < _freqsPerAxis; k++)
        {
            double exp = _freqsPerAxis == 1 ? 0.0 : (double)k / (_freqsPerAxis - 1);  // linspace(0,1,n)
            _freqBase[k] = Math.Pow(_theta, exp) * Math.PI / 2.0;
        }
    }

    /// <summary>Per-head dim is irrelevant here — RoPE spans the full inner dim.</summary>
    public int Dim => _dim;

    /// <summary>Builds cos/sin <c>[S, dim]</c> for a latent grid of <c>numFrames × height × width</c> tokens (row-major <c>(f,h,w)</c>). <paramref name="interpScale"/> scales each axis' coordinate (default 1 each).</summary>
    public (Tensor Cos, Tensor Sin) BuildCosSin(int numFrames, int height, int width, (double T, double H, double W) interpScale)
    {
        int seq = numFrames * height * width;
        Tensor cos = new Tensor(new TensorShape(seq, _dim), DType.F32);
        Tensor sin = new Tensor(new TensorShape(seq, _dim), DType.F32);
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;

        double normT = interpScale.T * _patchSizeT / _baseFrames;
        double normH = interpScale.H * _patchSize / _baseHeight;
        double normW = interpScale.W * _patchSize / _baseWidth;

        for (int fi = 0; fi < numFrames; fi++)
            for (int hi = 0; hi < height; hi++)
                for (int wi = 0; wi < width; wi++)
                {
                    long token = ((long)fi * height + hi) * width + wi;
                    // Axis phases (coord·2 − 1 after normalization).
                    double[] axisCoord = [fi * normT, hi * normH, wi * normW];
                    long baseOff = token * _dim;

                    // Identity pad at the front.
                    for (int d = 0; d < _pad; d++) { cp[baseOff + d] = 1f; sp[baseOff + d] = 0f; }

                    // freqs layout: for k in freqsPerAxis: for axis in {f,h,w}: value; then repeat_interleave(2).
                    int outIdx = _pad;
                    for (int k = 0; k < _freqsPerAxis; k++)
                        for (int axis = 0; axis < 3; axis++)
                        {
                            double phase = _freqBase[k] * (axisCoord[axis] * 2.0 - 1.0);
                            float c = (float)Math.Cos(phase);
                            float s = (float)Math.Sin(phase);
                            cp[baseOff + outIdx] = c; sp[baseOff + outIdx] = s;
                            cp[baseOff + outIdx + 1] = c; sp[baseOff + outIdx + 1] = s;
                            outIdx += 2;
                        }
                }

        return (cos, sin);
    }

    /// <summary>Applies the interleaved-pair rotation in-place to <paramref name="x"/> <c>[S, dim]</c> (the full-dim Q or K, before head split). cos/sin are <c>[S, dim]</c>. For each pair <c>(2j, 2j+1)</c>: <c>out[2j]=x[2j]·c − x[2j+1]·s</c>, <c>out[2j+1]=x[2j+1]·c + x[2j]·s</c>.</summary>
    public void ApplyRotary(IBackend backend, Tensor x, Tensor cos, Tensor sin)
    {
        int seq = (int)x.Shape[0];
        if ((int)x.Shape[1] != _dim)
            throw new ArgumentException($"x dim {x.Shape[1]} != rope dim {_dim}.", nameof(x));
        // Interleaved-pair rotation over the full inner dim (heads=1, headDim=dim). cos/sin use the duplicated-pair
        // layout (cos[2j]==cos[2j+1]), which is exactly what wan_rope_interleaved reads (coff = s·dim + 2i, same
        // rotation), so RoPE runs GPU-resident via the backend kernel instead of a host DataPointer loop. The host
        // loop had to D2H the [S,dim] Q/K, mutate on the CPU, then the next device op re-uploaded them — a big-tensor
        // cache miss per attention that was the #1 LTX-0.9 GPU cost (H2D_MISS_BIG in the sync profile).
        backend.WanRopeInterleaved(x, cos, sin, seq, 1, _dim);
    }
}
