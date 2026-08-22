using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>Hunyuan3D-2 shape DiT: a <b>Flux</b> transformer over a VecSet latent, image-conditioned via joint attention to DINOv2-giant tokens, with <b>no RoPE</b> since a VecSet is permutation-invariant.</summary>
public sealed unsafe class Hunyuan3DDit
{
    // ── CUDA-graph step capture. The 48-block DiT forward is per-op-host-overhead-bound (GEMM_F16 = 0%);
    //    capturing the device-resident block loop once and replaying it with a single cuGraphLaunch per step
    //    collapses ~720 host launches/forward → ~0. The timestep-varying modulation vector is computed OUTSIDE
    //    capture and CopyInto'd into _vecFixed (stable address), so the graph captures ONCE and replays all steps
    //    (latent/cond geometry is fixed across the loop). Clean eager fallback on any capture failure. ──
    private const int GraphCaptureCall = 3;
    private Tensor? _imgFixed, _txtFixed, _vecFixed, _velFixed;
    private long _graphSig = long.MinValue;
    private int _graphSigCalls;
    private bool _graphDead;

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

    /// <summary>Predicts the flow velocity for <paramref name="latent"/> <c>[B,N,C]</c> at <paramref name="timestep"/> ∈ [0,1], conditioned on DINOv2-giant tokens <paramref name="cond"/>.</summary>
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

        // ── Step-graph path: capture the 48-block loop + final layer once, replay per step. img/txt/vec (the only
        //    per-step-varying inputs, incl. the timestep) are refreshed into fixed buffers via CopyInto BEFORE each
        //    launch, so the captured region reads stable device addresses. Geometry (n, scond) is fixed across the
        //    denoise loop → captured once. Falls back to eager cleanly on any capture failure. ──
        bool graphMode = DitStepGraph.EnabledDefaultOn && backend.StepGraphSupported && !_graphDead;
        if (graphMode)
        {
            _imgFixed ??= new(new TensorShape(b, n, width), DType.F32);
            _txtFixed ??= new(new TensorShape(b, scond, width), DType.F32);
            _vecFixed ??= new(new TensorShape(b, width), DType.F32);
            _velFixed ??= new(new TensorShape(b, n, _cfg.LatentChannels), DType.F32);
            backend.CopyInto(_imgFixed, img); img.Dispose();
            backend.CopyInto(_txtFixed, txt); txt.Dispose();
            backend.CopyInto(_vecFixed, vec); vec.Dispose();

            long sig = ((long)n << 24) ^ scond;
            bool ownerLost = !ReferenceEquals(backend.StepGraphOwner, this);
            if (sig != _graphSig || ownerLost)
            {
                backend.StepGraphReset();
                _graphSig = sig; _graphSigCalls = 0; backend.StepGraphOwner = this;
            }
            _graphSigCalls++;

            if (backend.StepGraphReady && _graphSigCalls > GraphCaptureCall)
            {
                backend.StepGraphLaunch();
                return CopyOut(backend);
            }

            bool capture = _graphSigCalls == GraphCaptureCall;
            if (capture) backend.StepGraphBegin();
            try
            {
                RunBlocksIntoFixed(backend, n, scond);
            }
            catch (Exception ex) when (capture)
            {
                backend.StepGraphReset(); _graphDead = true;
                HartsyInference.Core.Logging.Logs.Warning($"[Hunyuan3D graph] capture invalidated — eager fallback: {ex.Message}");
                RunBlocksIntoFixed(backend, n, scond);
                return CopyOut(backend);
            }
            if (capture)
            {
                try { backend.StepGraphEndAndLaunch(); HartsyInference.Core.Logging.Logs.Info("[Hunyuan3D graph] DiT forward (48 blocks + final) captured; replaying via cuGraphLaunch."); }
                catch (Exception ex)
                {
                    backend.StepGraphReset(); _graphDead = true;
                    HartsyInference.Core.Logging.Logs.Warning($"[Hunyuan3D graph] capture failed — eager fallback: {ex.Message}");
                    RunBlocksIntoFixed(backend, n, scond);
                }
            }
            return CopyOut(backend);
        }

        Tensor velocity = RunBlocks(backend, img, txt, vec, n, scond, ownInputs: true);
        vec.Dispose();
        return velocity;
    }

    /// <summary>The device-resident block loop + final layer, shared by the eager and graph-capture paths; when <paramref name="ownInputs"/> is false the initial img/txt are fixed buffers not disposed on first use.</summary>
    private Tensor RunBlocks(IBackend backend, Tensor img, Tensor txt, Tensor vec, int n, int scond, bool ownInputs)
    {
        int b = (int)img.Shape[0], width = _cfg.Width;

        // F16 hot path (HARTSY_DIT_F16): one cast into F16 at the block-loop boundary — the double/single blocks
        // dtype-follow their input, so the whole loop runs in F16 (half the HBM traffic + F16 tensor-core GEMMs).
        // The timestep-modulation vec stays F32; the final layer casts back to F32 below. Matches Python fp16.
        DType act = DitDtype.Act;
        if (act == DType.F16)
        {
            Tensor imgF = new(img.Shape, DType.F16); backend.CastToF16(imgF, img);
            Tensor txtF = new(txt.Shape, DType.F16); backend.CastToF16(txtF, txt);
            if (ownInputs) { img.Dispose(); txt.Dispose(); }
            img = imgF; txt = txtF; ownInputs = true;
        }

        bool ownImg = ownInputs, ownTxt = ownInputs;
        foreach (Hunyuan3DDoubleBlock block in _double)
        {
            (Tensor ni, Tensor nt) = block.Forward(backend, img, txt, vec);
            if (ownImg) img.Dispose();
            if (ownTxt) txt.Dispose();
            img = ni; txt = nt; ownImg = ownTxt = true;
        }

        // Single stream over concat[txt, img] (txt FIRST); then drop the txt prefix (B=1 → contiguous rows).
        Tensor joint = new(new TensorShape(b, scond + n, width), img.DType); backend.Concat(joint, [txt, img], 1);
        img.Dispose(); txt.Dispose();
        foreach (Hunyuan3DSingleBlock block in _single)
        {
            Tensor nj = block.Forward(backend, joint, vec);
            joint.Dispose(); joint = nj;
        }
        Tensor latentOut = new(new TensorShape(b, n, width), joint.DType); backend.SliceRows(latentOut, joint, scond);
        joint.Dispose();
        if (latentOut.DType != DType.F32)
        {
            Tensor lo32 = new(latentOut.Shape, DType.F32); backend.CastToF32(lo32, latentOut); latentOut.Dispose(); latentOut = lo32;
        }

        // final_layer (LastLayer): shift,scale = chunk(adaLN(vec)) — SHIFT FIRST; x=(1+scale)·norm(x)+shift; linear→C.
        Tensor[] fm = Hunyuan3DGpuOps.ModParams(backend, vec, _finalNormW!, _finalNormB!, 2, width);   // fm[0]=shift, fm[1]=scale
        Tensor modOut = new(latentOut.Shape, DType.F32); backend.LayerNormModulate(modOut, latentOut, fm[1], fm[0], 1e-6f);
        latentOut.Dispose(); fm[0].Dispose(); fm[1].Dispose();
        Tensor velocity = new(new TensorShape(b, n, _cfg.LatentChannels), DType.F32);
        backend.Linear(velocity, modOut, _finalLinW!, _finalLinB!);
        modOut.Dispose();
        return velocity;
    }

    /// <summary>Runs the block loop over the fixed input buffers and lands the velocity in a stable, non-graph-owned buffer so replay's output survives the graph-memory auto-free.</summary>
    private void RunBlocksIntoFixed(IBackend backend, int n, int scond)
    {
        Tensor v = RunBlocks(backend, _imgFixed!, _txtFixed!, _vecFixed!, n, scond, ownInputs: false);
        backend.CopyInto(_velFixed!, v);
        v.Dispose();
    }

    /// <summary>Returns a fresh copy of the captured-graph velocity buffer so the caller may dispose it freely while the fixed buffer persists across steps.</summary>
    private Tensor CopyOut(IBackend backend)
    {
        Tensor o = new(_velFixed!.Shape, DType.F32);
        backend.CopyInto(o, _velFixed!);
        return o;
    }

    // hy3dgen passes time_factor as the max_period positional arg, so BOTH the t-scale and frequency base
    // are TimeFactor (1000) — NOT the usual 10000 default.
    private static void FluxTimestepEmbedding(Tensor outp, float timestep, int batch, int dim, float timeFactor)
    {
        float* p = (float*)outp.DataPointer;
        int half = dim / 2;
        float tt = timestep * timeFactor;
        float logMax = MathF.Log(timeFactor);   // max_period == time_factor (== 1000) in the hy3dgen call
        for (int bb = 0; bb < batch; bb++)
        {
            float* row = p + (long)bb * dim;
            for (int i = 0; i < half; i++)
            {
                float freq = MathF.Exp(-logMax * i / half);
                float a = tt * freq;
                row[i] = MathF.Cos(a);
                row[half + i] = MathF.Sin(a);
            }
        }
    }

    internal static Tensor F32(Tensor t) => t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}
