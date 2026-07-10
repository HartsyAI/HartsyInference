using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Multi-axis rotary position embedding for Z-Image (Lumina2/NextDiT). Splits the head dimension across 3 axes [frame, height, width] = [32, 48, 48] with θ=256. Caption tokens get position [0,0,0]; image tokens get [0, row, col]. Caption pad tokens and image pad tokens use the same axis-0=0 fallback so RoPE is well-defined for the entire concatenated sequence.</summary>
public sealed unsafe class ZImageRope
{
    private readonly int[] _axesDim;
    private readonly float _theta;
    private readonly int _headDim;

    private float[]? _cosCache;
    private float[]? _sinCache;
    private int _cachedSeqLen;

    // GPU cos/sin tables for the device RoPE path: [seqLen, headDim] with the pair angle at even index 2i
    // (the layout backend.WanRopeInterleaved reads). Uploaded once per Precompute as preloaded weights.
    private Tensor? _gpuCos, _gpuSin;
    private bool _gpuTablesDirty = true;

    /// <summary>Creates a Z-Image RoPE with the given axis dims and base.</summary>
    /// <param name="axesDim">Per-axis dimensions. Must sum to head_dim. Default [32, 48, 48] (sums to 128).</param>
    /// <param name="theta">Base frequency. Default 256 (Z-Image; much smaller than Flux's 10000).</param>
    public ZImageRope(int[]? axesDim = null, float theta = 256.0f)
    {
        _axesDim = axesDim ?? [32, 48, 48];
        _theta = theta;
        _headDim = 0;
        for (int i = 0; i < _axesDim.Length; i++)
            _headDim += _axesDim[i];
    }

    /// <summary>Total head dimension (sum of all axes, 128 for Z-Image).</summary>
    public int HeadDim => _headDim;

    /// <summary>Precomputes per-token cos/sin tables across all 3 axes for the given position IDs. Call before <see cref="Forward"/>.</summary>
    /// <param name="posIds">Position IDs [totalSeqLen, numAxes] as F32. For Z-Image, axes are [frame=0, row, col] for image tokens; all-zero for caption + pad tokens.</param>
    public void Precompute(Tensor posIds)
    {
        int totalSeqLen = (int)posIds.Shape[0];
        int numAxes = (int)posIds.Shape[1];
        int halfDim = _headDim / 2;

        _cosCache = new float[totalSeqLen * halfDim];
        _sinCache = new float[totalSeqLen * halfDim];
        _cachedSeqLen = totalSeqLen;

        _gpuTablesDirty = true;   // new tables → re-upload on next ApplyGpu

        float* posPtr = (float*)posIds.DataPointer;
        int freqOffset = 0;

        int maxPairs = 0;
        for (int a = 0; a < numAxes; a++)
            maxPairs = Math.Max(maxPairs, _axesDim[a] / 2);
        Span<double> omega = stackalloc double[maxPairs];

        for (int axis = 0; axis < numAxes; axis++)
        {
            int axisDim = _axesDim[axis];
            int numPairs = axisDim / 2;

            // omega[k] = 1 / (theta ^ (2k / axisDim))
            for (int k = 0; k < numPairs; k++)
            {
                double scale = (double)(2 * k) / axisDim;
                omega[k] = 1.0 / Math.Pow(_theta, scale);
            }

            for (int s = 0; s < totalSeqLen; s++)
            {
                double pos = posPtr[s * numAxes + axis];
                for (int k = 0; k < numPairs; k++)
                {
                    double angle = pos * omega[k];
                    int idx = s * halfDim + freqOffset + k;
                    _cosCache[idx] = (float)Math.Cos(angle);
                    _sinCache[idx] = (float)Math.Sin(angle);
                }
            }

            freqOffset += numPairs;
        }
    }

    /// <summary>Applies RoPE rotation to Q/K in-place when both share the same head count (full MHA). Q/K shape [B, numHeads, seqLen, headDim]. <see cref="Precompute"/> must have been called with matching seqLen.</summary>
    public void Forward(Tensor q, Tensor k, int batch, int numHeads, int seqLen)
    {
        ForwardSingle(q, batch, numHeads, seqLen);
        ForwardSingle(k, batch, numHeads, seqLen);
    }

    /// <summary>Applies RoPE rotation to a single tensor in-place. Tensor shape [B, numHeads, seqLen, headDim]. Used when Q and K have different head counts (GQA — call once per tensor with its own head count).</summary>
    public void ForwardSingle(Tensor t, int batch, int numHeads, int seqLen)
    {
        if (_cosCache == null || _sinCache == null)
            throw new InvalidOperationException("ZImageRope.Precompute must be called before Forward.");

        int halfDim = _headDim / 2;
        float* tPtr = (float*)t.DataPointer;

        fixed (float* cosPtr = _cosCache, sinPtr = _sinCache)
        {
            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        int vecOffset = ((b * numHeads + h) * seqLen + s) * _headDim;
                        int freqIdx = s * halfDim;
                        ApplyRotation(tPtr + vecOffset, cosPtr + freqIdx, sinPtr + freqIdx, halfDim);
                    }
                }
            }
        }
    }

    /// <summary>Device RoPE (B=1, the pipeline case): rotates Q and K in-place on the PRE-permute
    /// <c>[1, seqLen, numHeads, headDim]</c> layout via <see cref="IBackend.WanRopeInterleaved"/> — the same
    /// interleaved-pair rotation as <see cref="ApplyRotation"/> (adjacent pair (2i, 2i+1), angle shared across
    /// heads), bit-identical to the host path because RoPE indexes only (position, dim). This replaces the host
    /// <see cref="Forward"/> that D2H-round-tripped the full ~63 MB Q and K activations every block every step
    /// (the dominant residual Z-Image cost after the block GPU-residency rewrite). Mirrors
    /// <c>FluxRope.ApplyGpuGqa</c> (the Krea2 port). B&gt;1 callers keep the host <see cref="Forward"/>.</summary>
    public void ApplyGpu(IBackend backend, Tensor q, Tensor k, int numHeads)
    {
        if (_cosCache == null || _sinCache == null)
            throw new InvalidOperationException("ZImageRope.Precompute must be called before ApplyGpu.");
        int seqLen = (int)q.Shape[1];
        if (seqLen != _cachedSeqLen)
            throw new InvalidOperationException($"ZImageRope.ApplyGpu: tensor seqLen {seqLen} != precomputed {_cachedSeqLen}.");

        (Tensor cos, Tensor sin) = GetGpuTables(backend);
        backend.WanRopeInterleaved(q, cos, sin, seqLen, numHeads, _headDim);
        backend.WanRopeInterleaved(k, cos, sin, seqLen, numHeads, _headDim);
    }

    /// <summary>GQA variant of <see cref="ApplyGpu"/>: rotates Q with <paramref name="numHeads"/> and K with
    /// <paramref name="numKvHeads"/> (RoPE indexes only (position, dim) — head count is just layout), same
    /// pre-permute <c>[1, S, H, D]</c> contract. Used by the Lumina2 GPU-resident block path.</summary>
    public void ApplyGpuGqa(IBackend backend, Tensor q, Tensor k, int numHeads, int numKvHeads)
    {
        if (_cosCache == null || _sinCache == null)
            throw new InvalidOperationException("ZImageRope.Precompute must be called before ApplyGpuGqa.");
        int seqLen = (int)q.Shape[1];
        if (seqLen != _cachedSeqLen)
            throw new InvalidOperationException($"ZImageRope.ApplyGpuGqa: tensor seqLen {seqLen} != precomputed {_cachedSeqLen}.");

        (Tensor cos, Tensor sin) = GetGpuTables(backend);
        backend.WanRopeInterleaved(q, cos, sin, seqLen, numHeads, _headDim);
        backend.WanRopeInterleaved(k, cos, sin, seqLen, numKvHeads, _headDim);
    }

    /// <summary>Expands the host <c>[S, headDim/2]</c> caches to the kernel's <c>[S, headDim]</c> layout (pair
    /// angle stored at even index 2i; odd slots unused) and uploads them once as preloaded weights. Re-uploads
    /// only after a new <see cref="Precompute"/>.</summary>
    private (Tensor cos, Tensor sin) GetGpuTables(IBackend backend)
    {
        if (!_gpuTablesDirty)
            return (_gpuCos!, _gpuSin!);

        if (_gpuCos is not null)
        {
            backend.FreeWeights([_gpuCos, _gpuSin!]);
            _gpuCos.Dispose();
            _gpuSin!.Dispose();
        }

        int halfDim = _headDim / 2;
        Tensor cos = new Tensor(new TensorShape(_cachedSeqLen, _headDim), DType.F32);
        Tensor sin = new Tensor(new TensorShape(_cachedSeqLen, _headDim), DType.F32);
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;
        fixed (float* ch = _cosCache, sh = _sinCache)
        {
            for (int s = 0; s < _cachedSeqLen; s++)
            {
                long src = (long)s * halfDim;
                long dst = (long)s * _headDim;
                for (int i = 0; i < halfDim; i++)
                {
                    cp[dst + 2 * i] = ch[src + i];
                    sp[dst + 2 * i] = sh[src + i];
                    cp[dst + 2 * i + 1] = 0f;   // unused by the kernel
                    sp[dst + 2 * i + 1] = 0f;
                }
            }
        }
        backend.PreloadWeights([cos, sin]);
        _gpuCos = cos;
        _gpuSin = sin;
        _gpuTablesDirty = false;
        return (cos, sin);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyRotation(float* vec, float* cos, float* sin, int halfDim)
    {
        for (int i = 0; i < halfDim; i++)
        {
            float x0 = vec[2 * i];
            float x1 = vec[2 * i + 1];
            vec[2 * i] = cos[i] * x0 - sin[i] * x1;
            vec[2 * i + 1] = sin[i] * x0 + cos[i] * x1;
        }
    }

    /// <summary>Builds position IDs for Z-Image's main DiT in [image, caption] concat order. Mirrors diffusers' `patchify_and_embed`: caption uses pos_grid_size=(capPaddedLen,1,1) and pos_start=(1,0,0) — every caption slot (real + pad) gets frame index 1..capPaddedLen; image uses pos_start=(capPaddedLen+1, 0, 0) with grid (1, hPacked, wPacked) so image tokens share frame=capPaddedLen+1 and vary in (h,w). Image pad slots stay at (0,0,0). Returns shape [imgPaddedLen + capPaddedLen, 3] F32.</summary>
    /// <param name="capPaddedLen">Padded caption sequence length (real_cap + (-real_cap)%32). Used for BOTH the caption frame range and the image frame offset, matching diffusers' single _pad_with_ids call which uses the padded grid size.</param>
    /// <param name="hPacked">Image height in patch units (latent_h / patch_size).</param>
    /// <param name="wPacked">Image width in patch units (latent_w / patch_size).</param>
    /// <param name="imgPaddedLen">Padded image sequence length (real h*w padded up to multiple of 32 with x_pad_token).</param>
    public static Tensor BuildPositionIds(int capPaddedLen, int hPacked, int wPacked, int imgPaddedLen)
    {
        int imgRealLen = hPacked * wPacked;
        if (imgRealLen > imgPaddedLen)
            throw new ArgumentException($"imgRealLen={imgRealLen} cannot exceed imgPaddedLen={imgPaddedLen}", nameof(imgPaddedLen));

        int totalLen = imgPaddedLen + capPaddedLen;
        TensorShape shape = new TensorShape(totalLen, 3);
        Tensor posIds = new Tensor(shape, DType.F32);

        float* ptr = (float*)posIds.DataPointer;
        for (int i = 0; i < totalLen * 3; i++)
            ptr[i] = 0f;

        // [0 .. imgPaddedLen) — image tokens. Real tokens get (capPaddedLen+1, row, col); pad slots stay (0,0,0).
        float imgFrame = capPaddedLen + 1;
        for (int row = 0; row < hPacked; row++)
        {
            for (int col = 0; col < wPacked; col++)
            {
                int idx = row * wPacked + col;
                ptr[idx * 3 + 0] = imgFrame;
                ptr[idx * 3 + 1] = row;
                ptr[idx * 3 + 2] = col;
            }
        }

        // [imgPaddedLen .. imgPaddedLen+capPaddedLen) — caption tokens including pads. Frame runs 1..capPaddedLen, h=w=0.
        for (int t = 0; t < capPaddedLen; t++)
        {
            int idx = imgPaddedLen + t;
            ptr[idx * 3 + 0] = t + 1;
        }

        return posIds;
    }

    /// <summary>Builds caption-only position IDs for the context_refiner stack. Diffusers patchify_and_embed creates these with pos_grid_size=(capPaddedLen, 1, 1) and pos_start=(1,0,0), so EVERY caption slot (real + pad) gets a unique frame index 1..capPaddedLen with h=w=0. Returns shape [capPaddedLen, 3] F32.</summary>
    public static Tensor BuildCaptionPositionIds(int capPaddedLen)
    {
        TensorShape shape = new TensorShape(capPaddedLen, 3);
        Tensor posIds = new Tensor(shape, DType.F32);
        float* ptr = (float*)posIds.DataPointer;
        for (int t = 0; t < capPaddedLen; t++)
        {
            ptr[t * 3 + 0] = t + 1;
            ptr[t * 3 + 1] = 0f;
            ptr[t * 3 + 2] = 0f;
        }
        return posIds;
    }

    /// <summary>Builds image-only position IDs for the noise_refiner stack (operates on padded image tokens with no caption present). Real tokens: (1, row, col); pad: (0,0,0). The frame=1 base mirrors diffusers' use of pos_start=(1,0,0) for image-only refinement.</summary>
    public static Tensor BuildImagePositionIds(int hPacked, int wPacked, int imgPaddedLen)
    {
        int imgRealLen = hPacked * wPacked;
        TensorShape shape = new TensorShape(imgPaddedLen, 3);
        Tensor posIds = new Tensor(shape, DType.F32);

        float* ptr = (float*)posIds.DataPointer;
        for (int i = 0; i < imgPaddedLen * 3; i++)
            ptr[i] = 0f;

        for (int row = 0; row < hPacked; row++)
        {
            for (int col = 0; col < wPacked; col++)
            {
                int idx = row * wPacked + col;
                ptr[idx * 3 + 0] = 1f;
                ptr[idx * 3 + 1] = row;
                ptr[idx * 3 + 2] = col;
            }
        }
        return posIds;
    }
}
