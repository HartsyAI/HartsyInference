using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>Hunyuan3D-2 VecSet ShapeVAE decoder (decode-only): a self-attention transformer over the latent set, then a <see cref="Hunyuan3DGeoDecoder"/> that cross-attends 3D query points to the latents for a dense occupancy grid.</summary>
public sealed unsafe class Hunyuan3DShapeVae
{
    private readonly Hunyuan3DConfig _cfg;
    private readonly Hunyuan3DVaeResBlock[] _resblocks;
    private readonly Hunyuan3DGeoDecoder _geo;
    private Tensor? _postKlW, _postKlB;
    private readonly int _fourierDim;

    public Hunyuan3DConfig Config => _cfg;
    public int FourierDim => _fourierDim;

    public Hunyuan3DShapeVae(Hunyuan3DConfig cfg)
    {
        _cfg = cfg;
        _resblocks = new Hunyuan3DVaeResBlock[cfg.VaeDepth];
        for (int i = 0; i < cfg.VaeDepth; i++) _resblocks[i] = new Hunyuan3DVaeResBlock(cfg.VaeWidth, cfg.VaeNumHeads);
        _fourierDim = 3 * (2 * cfg.FourierBands + 1);   // include_input + sin/cos: 3·(1+2·8) = 51
        _geo = new Hunyuan3DGeoDecoder(cfg.VaeWidth, cfg.VaeNumHeads, cfg.FourierBands, _fourierDim);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _postKlW = F32(w[$"{p}post_kl.weight"]); _postKlB = F32(w[$"{p}post_kl.bias"]);
        for (int i = 0; i < _resblocks.Length; i++) _resblocks[i].LoadWeights(w, $"{p}transformer.resblocks.{i}");
        _geo.LoadWeights(w, $"{p}geo_decoder");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_postKlW is not null) yield return _postKlW;
        if (_postKlB is not null) yield return _postKlB;
        foreach (Hunyuan3DVaeResBlock b in _resblocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (Tensor t in _geo.EnumerateWeights()) yield return t;
    }

    /// <summary>post_kl + transformer self-attention stack over the latent set. Returns <c>[1,N,Width]</c>.</summary>
    public Tensor ProcessLatents(IBackend backend, Tensor latent)
    {
        int b = (int)latent.Shape[0], n = (int)latent.Shape[1];
        Tensor x = new(new TensorShape(b, n, _cfg.VaeWidth), DType.F32);
        backend.Linear(x, latent, _postKlW!, _postKlB!);
        foreach (Hunyuan3DVaeResBlock block in _resblocks) { Tensor nx = block.Forward(backend, x); x.Dispose(); x = nx; }
        return x;
    }

    /// <summary>Precomputes the query-independent cross-attn K/V from processed latents; pair with <see cref="DecodePoints"/> and dispose both returned tensors.</summary>
    public (Tensor kNorm, Tensor v) PrepareKv(IBackend backend, Tensor processed) => _geo.PrepareKv(backend, processed);

    /// <summary>Occupancy logits for query points <c>[count,3]</c> given precomputed cross-attn K/V. Returns <c>[count]</c>.</summary>
    public Tensor DecodePoints(IBackend backend, (Tensor kNorm, Tensor v) kv, Tensor coords, int count)
        => _geo.Query(backend, kv, coords, count);

    /// <summary>Full grid decode: process latents once, then chunk the <paramref name="resolution"/>³ query grid over <c>[-bound,bound]</c> into occupancy → <see cref="ScalarField3D"/>.</summary>
    // chunkSize bounds the geo-decoder cross-attn scores buffer (H·chunk·Nlatent floats): 4096 → ~1.6 GB at
    // heads 16 / 3072 latents. Larger values (e.g. 32768 → ~6 GB) OOM alongside the weights on a 24 GB card.
    public ScalarField3D Decode(IBackend backend, Tensor latent, int resolution, float bound, int chunkSize = 4096)
    {
        if (latent.Shape.Rank != 3 || latent.Shape[0] != 1)
            throw new ArgumentException($"latent must be [1,N,C]; got {latent.Shape}.", nameof(latent));
        // Larger chunks = fewer per-chunk host stream-drains (the occ.DataPointer readback). The old 4096 bound the
        // MATERIALIZED cross-attn scores buffer, but the geo-decoder SDPA runs cuDNN fused-flash (allowF16), so no
        // scores materialize — the chunk is bounded only by activation memory. HARTSY_HY3D_VAE_CHUNK overrides.
        if (int.TryParse(Environment.GetEnvironmentVariable("HARTSY_HY3D_VAE_CHUNK"), out int envChunk) && envChunk > 0)
            chunkSize = envChunk;

        Tensor processed = ProcessLatents(backend, latent);
        (Tensor kNorm, Tensor v) kv = _geo.PrepareKv(backend, processed);
        processed.Dispose();

        int res = resolution;
        long total = (long)res * res * res;
        float[] values = new float[total];
        float step = res > 1 ? 2f * bound / (res - 1) : 0f;
        float[] coordBuf = new float[chunkSize * 3];

        long done = 0;
        while (done < total)
        {
            int count = (int)Math.Min(chunkSize, total - done);
            for (int i = 0; i < count; i++)
            {
                // ix fastest so values[idx] aligns with ScalarField3D.Index(x,y,z) = x + Res·(y + Res·z).
                long idx = done + i;
                int ix = (int)(idx % res); long tt = idx / res;
                int iy = (int)(tt % res); int iz = (int)(tt / res);
                coordBuf[i * 3 + 0] = -bound + ix * step;
                coordBuf[i * 3 + 1] = -bound + iy * step;
                coordBuf[i * 3 + 2] = -bound + iz * step;
            }
            // Host coords → device tensor (uploaded once per chunk); FourierEmbed then runs on-device.
            Tensor coordsT = new(new TensorShape(count, 3), DType.F32);
            new ReadOnlySpan<float>(coordBuf, 0, count * 3).CopyTo(new Span<float>((float*)coordsT.DataPointer, count * 3));
            Tensor occ = DecodePoints(backend, kv, coordsT, count);
            Tensor occF = occ.DType == DType.F32 ? occ : occ.CastTo(DType.F32);
            new ReadOnlySpan<float>((float*)occF.DataPointer, count).CopyTo(values.AsSpan((int)done, count));
            if (!ReferenceEquals(occF, occ)) occF.Dispose();
            occ.Dispose();
            coordsT.Dispose();   // after occ readback (which drains the stream) → coords upload safely complete
            done += count;
        }
        kv.kNorm.Dispose(); kv.v.Dispose();

        return new ScalarField3D
        {
            Values = values, ResX = res, ResY = res, ResZ = res,
            Min = (-bound, -bound, -bound), Max = (bound, bound, bound),
        };
    }

    internal static Tensor F32(Tensor t) => t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}
