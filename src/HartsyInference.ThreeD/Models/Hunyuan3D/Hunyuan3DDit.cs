using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>Hunyuan3D-2 shape DiT — a <b>Flux</b> transformer over a VecSet latent (N set tokens × C channels),
/// image-conditioned via cross-stream joint attention to DINOv2-giant tokens. Mirrors
/// <c>hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py</c>: <c>latent_in</c> → 16 <see cref="Hunyuan3DDoubleBlock"/>
/// (img=latent, txt=cond) → concat[cond,latent] → 32 <see cref="Hunyuan3DSingleBlock"/> → drop cond → LastLayer.
/// <b>No RoPE</b> (<c>pe=None</c>; a VecSet is permutation-invariant). Predicts the rectified-flow velocity.
/// Reuses the verified Flux helpers (<see cref="AdaLNModulation"/>, <see cref="QkNorm"/>, <see cref="SwiGluFfn"/>).</summary>
public sealed unsafe class Hunyuan3DDit
{
    private readonly Hunyuan3DConfig _cfg;
    private readonly Hunyuan3DDoubleBlock[] _double;
    private readonly Hunyuan3DSingleBlock[] _single;

    private Tensor? _latentInW, _latentInB;   // in_channels → Width
    private Tensor? _condInW, _condInB;        // CondDim → Width
    private Tensor? _timeIn1W, _timeIn1B;      // 256 → Width  (MLPEmbedder in_layer)
    private Tensor? _timeIn2W, _timeIn2B;      // Width → Width (MLPEmbedder out_layer)
    private Tensor? _finalNormW;               // final_layer.adaLN_modulation.1 (Width → 2*Width)
    private Tensor? _finalNormB;
    private Tensor? _finalLinW, _finalLinB;    // final_layer.linear (Width → C)

    public Hunyuan3DConfig Config => _cfg;

    public Hunyuan3DDit(Hunyuan3DConfig cfg)
    {
        _cfg = cfg;
        _double = new Hunyuan3DDoubleBlock[cfg.DepthDouble];
        for (int i = 0; i < cfg.DepthDouble; i++) _double[i] = new Hunyuan3DDoubleBlock(cfg.Width, cfg.NumHeads, cfg.MlpDim);
        _single = new Hunyuan3DSingleBlock[cfg.DepthSingle];
        for (int i = 0; i < cfg.DepthSingle; i++) _single[i] = new Hunyuan3DSingleBlock(cfg.Width, cfg.NumHeads, cfg.MlpDim);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _latentInW = F32(w[$"{p}latent_in.weight"]); _latentInB = F32(w[$"{p}latent_in.bias"]);
        _condInW = F32(w[$"{p}cond_in.weight"]); _condInB = F32(w[$"{p}cond_in.bias"]);
        _timeIn1W = F32(w[$"{p}time_in.in_layer.weight"]); _timeIn1B = F32(w[$"{p}time_in.in_layer.bias"]);
        _timeIn2W = F32(w[$"{p}time_in.out_layer.weight"]); _timeIn2B = F32(w[$"{p}time_in.out_layer.bias"]);
        for (int i = 0; i < _double.Length; i++) _double[i].LoadWeights(w, $"{p}double_blocks.{i}");
        for (int i = 0; i < _single.Length; i++) _single[i].LoadWeights(w, $"{p}single_blocks.{i}");
        _finalNormW = F32(w[$"{p}final_layer.adaLN_modulation.1.weight"]); _finalNormB = F32(w[$"{p}final_layer.adaLN_modulation.1.bias"]);
        _finalLinW = F32(w[$"{p}final_layer.linear.weight"]); _finalLinB = F32(w[$"{p}final_layer.linear.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] head = [_latentInW, _latentInB, _condInW, _condInB, _timeIn1W, _timeIn1B, _timeIn2W, _timeIn2B,
            _finalNormW, _finalNormB, _finalLinW, _finalLinB];
        foreach (Tensor? t in head) if (t is not null) yield return t;
        foreach (Hunyuan3DDoubleBlock b in _double) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (Hunyuan3DSingleBlock b in _single) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Predicts the flow velocity for <paramref name="latent"/> <c>[B,N,C]</c> at
    /// <paramref name="timestep"/> ∈ [0,1], conditioned on DINOv2-giant tokens <paramref name="cond"/>
    /// <c>[B, Scond, CondDim]</c>. Returns velocity <c>[B,N,C]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor cond, float timestep)
    {
        int b = (int)latent.Shape[0], n = (int)latent.Shape[1], width = _cfg.Width;

        // latent_in: C → Width.
        Tensor img = new(new TensorShape(b, n, width), DType.F32);
        backend.Linear(img, latent, _latentInW!, _latentInB!);

        // cond_in: CondDim → Width.
        int scond = (int)cond.Shape[1];
        Tensor txt = new(new TensorShape(b, scond, width), DType.F32);
        backend.Linear(txt, cond, _condInW!, _condInB!);

        // time_in: MLPEmbedder(sinusoid(t*time_factor, 256)) → SiLU → Linear → [B, Width].
        Tensor tSin = new(new TensorShape(b, _cfg.TimestepEmbedDim), DType.F32);
        FluxTimestepEmbedding(tSin, timestep, b, _cfg.TimestepEmbedDim, _cfg.TimeFactor);
        Tensor t1 = new(new TensorShape(b, width), DType.F32); backend.Linear(t1, tSin, _timeIn1W!, _timeIn1B!); tSin.Dispose();
        Tensor t1a = new(t1.Shape, DType.F32); backend.Silu(t1a, t1); t1.Dispose();
        Tensor vec = new(new TensorShape(b, width), DType.F32); backend.Linear(vec, t1a, _timeIn2W!, _timeIn2B!); t1a.Dispose();

        // Double stream: (img, txt) updated jointly.
        foreach (Hunyuan3DDoubleBlock block in _double)
        {
            (Tensor ni, Tensor nt) = block.Forward(backend, img, txt, vec);
            img.Dispose(); txt.Dispose(); img = ni; txt = nt;
        }

        // Single stream over concat[txt, img] (txt FIRST); then drop the txt prefix.
        Tensor joint = Hunyuan3DDitOps.ConcatSeq(txt, img, b, scond, n, width);
        txt.Dispose(); img.Dispose();
        foreach (Hunyuan3DSingleBlock block in _single)
        {
            Tensor nj = block.Forward(backend, joint, vec);
            joint.Dispose(); joint = nj;
        }
        Tensor latentOut = Hunyuan3DDitOps.SliceSeq(joint, b, scond, n, width);   // drop first `scond` tokens
        joint.Dispose();

        // final_layer (LastLayer): shift,scale = chunk(adaLN(vec)) — SHIFT FIRST; x=(1+scale)·norm(x)+shift; linear→C.
        Tensor tAct = new(vec.Shape, DType.F32); backend.Silu(tAct, vec); vec.Dispose();
        Tensor mod = new(new TensorShape(b, 2 * width), DType.F32); backend.Linear(mod, tAct, _finalNormW!, _finalNormB!); tAct.Dispose();
        Tensor normed = new(latentOut.Shape, DType.F32);
        Hunyuan3DDitOps.LayerNormNoAffine(normed, latentOut, b, n, width);
        latentOut.Dispose();
        Hunyuan3DDitOps.ModulateShiftFirst(normed, mod, b, n, width);   // shift=param0, scale=param1
        mod.Dispose();
        Tensor velocity = new(new TensorShape(b, n, _cfg.LatentChannels), DType.F32);
        backend.Linear(velocity, normed, _finalLinW!, _finalLinB!);
        normed.Dispose();
        return velocity;
    }

    /// <summary>Flux sinusoidal timestep embedding: <c>t*=time_factor</c>, half=dim/2, <c>freqs=exp(-log(10000)·i/half)</c>,
    /// <c>emb=[cos(t·freqs), sin(t·freqs)]</c> (cos first). Same scalar t across the batch.</summary>
    private static void FluxTimestepEmbedding(Tensor outp, float timestep, int batch, int dim, float timeFactor)
    {
        float* p = (float*)outp.DataPointer;
        int half = dim / 2;
        float tt = timestep * timeFactor;
        for (int bb = 0; bb < batch; bb++)
        {
            float* row = p + (long)bb * dim;
            for (int i = 0; i < half; i++)
            {
                float freq = MathF.Exp(-MathF.Log(10000f) * i / half);
                float a = tt * freq;
                row[i] = MathF.Cos(a);
                row[half + i] = MathF.Sin(a);
            }
        }
    }

    internal static Tensor F32(Tensor t) => t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}
