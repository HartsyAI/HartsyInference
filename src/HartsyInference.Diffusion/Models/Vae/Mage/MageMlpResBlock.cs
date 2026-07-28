using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae.Mage;

/// <summary>A dec_net residual block from MageVAE's per-patch NeRF decoder (mage_vae.py <c>SimpleMLPAdaLN._MLPResBlock</c>,
/// L245). Operates per sub-pixel on <c>[rows, 256, 32]</c> with a per-position AdaLN conditioning <c>y</c> (from
/// <c>cond_embed(s_tokens)</c>). Forward: <c>shift,scale,gate = adaLN(y).chunk(3)</c>; <c>h = in_ln(x)·(1+scale)+shift</c>;
/// <c>return x + gate·mlp(h)</c>, where <c>mlp = Linear(32→32)→SiLU→Linear(32→32)</c>. Unlike the DiCoBlock's
/// timestep-only modulation, this AdaLN depends on the per-position latent, so it is computed every forward.</summary>
public sealed unsafe class MageMlpResBlock
{
    private readonly int _c;   // 32
    private Tensor? _inLnW, _inLnB;      // in_ln (LayerNorm 32; affine — falls back to identity if absent)
    private Tensor? _mlp0W, _mlp0B, _mlp2W, _mlp2B;
    private Tensor? _adaW, _adaB;        // adaLN_modulation.1 (Linear 32→96)

    public MageMlpResBlock(int channels) => _c = channels;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _inLnW = w.TryGetValue($"{p}.in_ln.weight", out Tensor? lw) ? F32(lw) : Ones(_c);
        _inLnB = w.TryGetValue($"{p}.in_ln.bias", out Tensor? lb) ? F32(lb) : Zeros(_c);
        _mlp0W = F32(w[$"{p}.mlp.0.weight"]); _mlp0B = F32(w[$"{p}.mlp.0.bias"]);
        _mlp2W = F32(w[$"{p}.mlp.2.weight"]); _mlp2B = F32(w[$"{p}.mlp.2.bias"]);
        _adaW = F32(w[$"{p}.adaLN_modulation.1.weight"]); _adaB = F32(w[$"{p}.adaLN_modulation.1.bias"]);
    }

    private static Tensor F32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); new Span<float>((float*)t.DataPointer, n).Fill(1f); return t; }
    private static Tensor Zeros(int n) { Tensor t = new(new TensorShape(n), DType.F32); new Span<float>((float*)t.DataPointer, n).Clear(); return t; }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _inLnW, _inLnB, _mlp0W, _mlp0B, _mlp2W, _mlp2B, _adaW, _adaB })
            if (t is not null) yield return t;
    }

    // x [rows,256,32], y [rows,256,32]. Returns [rows,256,32].
    public Tensor Forward(IBackend backend, Tensor x, Tensor y, int rows)
    {
        int n = rows * 256, c = _c;
        Tensor xFlat = x.Reshape(new TensorShape(n, c));
        Tensor yFlat = y.Reshape(new TensorShape(n, c));

        // adaLN = Linear(silu(y)) → [n, 3c]; chunk → shift/scale/gate.
        Tensor ySilu = new(new TensorShape(n, c), DType.F32);
        backend.Silu(ySilu, yFlat);
        Tensor mod = new(new TensorShape(n, 3 * c), DType.F32);
        backend.Linear(mod, ySilu, _adaW!, _adaB); ySilu.Dispose();

        // h = in_ln(x) then modulate per-element: h*(1+scale)+shift.
        Tensor h = new(new TensorShape(n, c), DType.F32);
        backend.LayerNorm(h, xFlat, _inLnW!, _inLnB!, 1e-6f);
        float* hp = (float*)h.DataPointer; float* mp = (float*)mod.DataPointer;
        for (long i = 0; i < n; i++)
            for (int k = 0; k < c; k++)
            {
                float shift = mp[i * 3 * c + k];
                float scale = mp[i * 3 * c + c + k];
                hp[i * c + k] = hp[i * c + k] * (1f + scale) + shift;
            }

        // mlp(h) = Linear→SiLU→Linear.
        Tensor m0 = new(new TensorShape(n, c), DType.F32);
        backend.Linear(m0, h, _mlp0W!, _mlp0B); h.Dispose();
        Tensor m0a = new(new TensorShape(n, c), DType.F32);
        backend.Silu(m0a, m0); m0.Dispose();
        Tensor m2 = new(new TensorShape(n, c), DType.F32);
        backend.Linear(m2, m0a, _mlp2W!, _mlp2B); m0a.Dispose();

        // out = x + gate * mlp.
        Tensor outp = new(new TensorShape(n, c), DType.F32);
        float* op = (float*)outp.DataPointer; float* xp = (float*)xFlat.DataPointer; float* m2p = (float*)m2.DataPointer;
        for (long i = 0; i < n; i++)
            for (int k = 0; k < c; k++)
            {
                float gate = mp[i * 3 * c + 2 * c + k];
                op[i * c + k] = xp[i * c + k] + gate * m2p[i * c + k];
            }
        mod.Dispose(); m2.Dispose();
        return outp.Reshape(new TensorShape(rows, 256, c));
    }
}
