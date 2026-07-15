using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Matrix-Game 3.0 dual-stream action conditioning (<c>wan/modules/action_module.py</c>) — the per-block side
/// branch attached to a subset of the Wan2.2 DiT blocks. Two streams over the latent-frame axis, evaluated per spatial
/// position: <b>mouse</b> (continuous Δx/Δy: window-grouped raw actions concatenated with the image token → MLP →
/// temporal self-attention with its own 3D RoPE <c>[8,28,28]</c> θ=256) and <b>keyboard</b> (discrete one-hot: embed →
/// window-group → K/V cross-attended by an image-token query). Both project back to the DiT width and add residually.
/// Raw actions arrive at the RGB frame rate; each latent frame gathers a window of
/// <c>vae_ratio · windows_size = 12</c> raw actions (left-padded by repeating the first frame). Memory frames (no
/// recorded actions in v1) get zero windows; RoPE uses the caller's per-frame indices so memory keeps historical
/// positions. Numerics are validation-pending against the reference.</summary>
public sealed unsafe class MatrixGame3ActionModule
{
    private readonly int _imgDim;            // DiT hidden width
    private readonly int _hiddenSize;        // keyboard embed width (128)
    private readonly int _mouseDim;          // raw mouse floats per frame (2)
    private readonly int _keyboardDim;       // raw keyboard floats per frame (6)
    private readonly int _streamDim;         // attention width per stream (1024)
    private readonly int _heads;             // 16
    private readonly int _headDim;
    private readonly int _ratio;             // VAE temporal compression (4)
    private readonly int _window;            // windows_size (3)
    private readonly bool _enableMouse;
    private readonly bool _enableKeyboard;
    private readonly WanRope _rope;
    private readonly float _eps;

    private Tensor? _mouseMlp1W, _mouseMlp1B, _mouseMlp2W, _mouseMlp2B, _mouseLnW, _mouseLnB;
    private Tensor? _tQkvW, _tQkvB, _projMouseW, _projMouseB;
    private Tensor? _imgQNorm, _imgKNorm;
    private Tensor? _kbdEmbed1W, _kbdEmbed1B, _kbdEmbed2W, _kbdEmbed2B;
    private Tensor? _kbdKvW, _kbdKvB, _kbdQW, _kbdQB, _projKbdW, _projKbdB;
    private Tensor? _keyQNorm, _keyKNorm;

    public MatrixGame3ActionModule(int imgHiddenSize, int mouseDimIn = 2, int keyboardDimIn = 6, int hiddenSize = 128,
        int streamHiddenDim = 1024, int headsNum = 16, int vaeTimeCompressionRatio = 4, int windowsSize = 3,
        double ropeTheta = 256, int ropeTDim = 8, int ropeHDim = 28, int ropeWDim = 28, float eps = 1e-6f,
        bool enableMouse = true, bool enableKeyboard = true)
    {
        _imgDim = imgHiddenSize;
        _mouseDim = mouseDimIn;
        _keyboardDim = keyboardDimIn;
        _hiddenSize = hiddenSize;
        _streamDim = streamHiddenDim;
        _heads = headsNum;
        _headDim = streamHiddenDim / headsNum;
        _ratio = vaeTimeCompressionRatio;
        _window = windowsSize;
        _enableMouse = enableMouse;
        _enableKeyboard = enableKeyboard;
        _eps = eps;
        if (ropeTDim + ropeHDim + ropeWDim != _headDim)
            throw new ArgumentException($"rope dims ({ropeTDim}+{ropeHDim}+{ropeWDim}) must sum to head dim {_headDim}.");
        _rope = new WanRope(ropeTDim, ropeHDim, ropeWDim, ropeTheta);
    }

    /// <summary>Raw actions gathered per latent frame.</summary>
    public int WindowLength => _ratio * _window;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        if (_enableMouse)
        {
            _mouseMlp1W = w[$"{prefix}.mouse_mlp.0.weight"]; w.TryGetValue($"{prefix}.mouse_mlp.0.bias", out _mouseMlp1B);
            _mouseMlp2W = w[$"{prefix}.mouse_mlp.2.weight"]; w.TryGetValue($"{prefix}.mouse_mlp.2.bias", out _mouseMlp2B);
            // Both LayerNorm affine tensors must be F32: AffineBroadcastLastDim rejects a bf16 scale/shift (the bias was
            // loaded as-is → bf16 → NotSupported on the GPU action path; the CPU parity run cast everything to F32 first).
            _mouseLnW = LoadF32(w, $"{prefix}.mouse_mlp.3.weight"); _mouseLnB = LoadF32Opt(w, $"{prefix}.mouse_mlp.3.bias");
            _tQkvW = w[$"{prefix}.t_qkv.weight"]; w.TryGetValue($"{prefix}.t_qkv.bias", out _tQkvB);
            _projMouseW = w[$"{prefix}.proj_mouse.weight"]; w.TryGetValue($"{prefix}.proj_mouse.bias", out _projMouseB);
            _imgQNorm = LoadF32(w, $"{prefix}.img_attn_q_norm.weight");
            _imgKNorm = LoadF32(w, $"{prefix}.img_attn_k_norm.weight");
        }
        if (_enableKeyboard)
        {
            _kbdEmbed1W = w[$"{prefix}.keyboard_embed.0.weight"]; w.TryGetValue($"{prefix}.keyboard_embed.0.bias", out _kbdEmbed1B);
            _kbdEmbed2W = w[$"{prefix}.keyboard_embed.2.weight"]; w.TryGetValue($"{prefix}.keyboard_embed.2.bias", out _kbdEmbed2B);
            _kbdKvW = w[$"{prefix}.keyboard_attn_kv.weight"]; w.TryGetValue($"{prefix}.keyboard_attn_kv.bias", out _kbdKvB);
            _kbdQW = w[$"{prefix}.mouse_attn_q.weight"]; w.TryGetValue($"{prefix}.mouse_attn_q.bias", out _kbdQB);
            _projKbdW = w[$"{prefix}.proj_keyboard.weight"]; w.TryGetValue($"{prefix}.proj_keyboard.bias", out _projKbdB);
            _keyQNorm = LoadF32(w, $"{prefix}.key_attn_q_norm.weight");
            _keyKNorm = LoadF32(w, $"{prefix}.key_attn_k_norm.weight");
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _mouseMlp1W, _mouseMlp1B, _mouseMlp2W, _mouseMlp2B, _mouseLnW, _mouseLnB,
            _tQkvW, _tQkvB, _projMouseW, _projMouseB, _imgQNorm, _imgKNorm,
            _kbdEmbed1W, _kbdEmbed1B, _kbdEmbed2W, _kbdEmbed2B, _kbdKvW, _kbdKvB, _kbdQW, _kbdQB,
            _projKbdW, _projKbdB, _keyQNorm, _keyKNorm })
            if (t is not null) yield return t;
    }

    /// <summary>Adds the action conditioning residual to <paramref name="hidden"/> <c>[S, dim]</c> in place.
    /// <paramref name="grid"/> is the token grid (frames, h, w) with frames covering memory+past+current;
    /// <paramref name="frameIndices"/> are the RoPE t-positions per frame (memory keeps historical indices);
    /// <paramref name="mouse"/>/<paramref name="keyboard"/> are RGB-rate rows for the NON-memory frames (the trailing
    /// <c>grid.Frames − memoryFrames</c> latent frames); memory frames get zero action windows.</summary>
    public void Forward(IBackend backend, Tensor hidden, (int Frames, int H, int W) grid, ReadOnlySpan<int> frameIndices,
        int memoryFrames, Tensor mouse, Tensor keyboard)
    {
        int tt = grid.Frames, sp = grid.H * grid.W;
        if (frameIndices.Length != tt)
            throw new ArgumentException($"frameIndices must have {tt} entries.", nameof(frameIndices));
        if ((int)hidden.Shape[0] != tt * sp)
            throw new ArgumentException($"hidden rows {hidden.Shape[0]} != grid tokens {tt * sp}.", nameof(hidden));

        if (_enableMouse)
        {
            float[] mouseWin = BuildWindows(mouse, _mouseDim, tt, memoryFrames);      // [tt, window·mouseDim]
            MouseStream(backend, hidden, tt, grid.H, grid.W, frameIndices, mouseWin);
        }
        if (_enableKeyboard)
            KeyboardStream(backend, hidden, tt, grid.H, grid.W, frameIndices, memoryFrames, keyboard);
    }

    /// <summary>Gathers the per-latent-frame action window: latent frame <c>i</c> (0-based among non-memory frames)
    /// covers raw frames <c>[i·ratio − ratio·window, i·ratio)</c> — 12 raw rows ending just before the frame's raw
    /// time — left-padded by repeating row 0. Mirrors upstream's <c>pad_t = ratio·window</c> front-pad then slice
    /// <c>padded[ratio·i : ratio·i + pad_t]</c> (action_module.py). Memory frames produce zero windows.</summary>
    private float[] BuildWindows(Tensor raw, int dim, int tt, int memoryFrames)
    {
        int winLen = WindowLength;
        int rawFrames = (int)raw.Shape[0];
        float* rp = (float*)raw.DataPointer;
        float[] result = new float[tt * winLen * dim];
        for (int f = memoryFrames; f < tt; f++)
        {
            int local = f - memoryFrames;
            int start = local * _ratio - _ratio * _window;
            for (int k = 0; k < winLen; k++)
            {
                int src = Math.Clamp(start + k, 0, rawFrames - 1);
                for (int d = 0; d < dim; d++)
                    result[((long)f * winLen + k) * dim + d] = rp[(long)src * dim + d];
            }
        }
        return result;
    }

    private void MouseStream(IBackend backend, Tensor hidden, int tt, int gh, int gw, ReadOnlySpan<int> frameIndices, float[] mouseWin)
    {
        int sp = gh * gw, winFloats = WindowLength * _mouseDim;

        // Per-token MLP input: [token features (dim) | grouped mouse window (winFloats)] — sequence axis is the frame.
        // Device concat (the old host loop read the [S, dim] hidden D2H every mouse block). The tiny window array is
        // uploaded once into a device tensor.
        Tensor mouseWinT = new Tensor(new TensorShape(tt, winFloats), DType.F32);
        fixed (float* mw = mouseWin)
            Buffer.MemoryCopy(mw, (float*)mouseWinT.DataPointer, (long)tt * winFloats * 4, (long)tt * winFloats * 4);
        Tensor mlpIn = new Tensor(new TensorShape(tt * sp, _imgDim + winFloats), DType.F32);
        backend.Mg3MouseMlpConcat(mlpIn, hidden, mouseWinT, tt, sp, _imgDim, winFloats);
        mouseWinT.Dispose();

        Tensor h1 = new Tensor(new TensorShape(tt * sp, _streamDim), DType.F32);
        backend.Linear(h1, mlpIn, _mouseMlp1W!, _mouseMlp1B);
        mlpIn.Dispose();
        Tensor act = new Tensor(h1.Shape, DType.F32); backend.Gelu(act, h1); h1.Dispose();
        Tensor h2 = new Tensor(new TensorShape(tt * sp, _streamDim), DType.F32);
        backend.Linear(h2, act, _mouseMlp2W!, _mouseMlp2B);
        act.Dispose();
        LayerNormRows(backend, h2, _mouseLnW!, _mouseLnB, tt * sp, _streamDim);

        Tensor qkv = new Tensor(new TensorShape(tt * sp, 3 * _streamDim), DType.F32);
        backend.Linear(qkv, h2, _tQkvW!, _tQkvB);
        h2.Dispose();

        // Temporal self-attention per spatial position, batched as [sp, heads, tt, headDim].
        (Tensor cos, Tensor sin) = _rope.BuildCosSin(frameIndices, gh, gw);
        Tensor q = SplitQkvTemporal(backend, qkv, tt, sp, 0);
        Tensor k = SplitQkvTemporal(backend, qkv, tt, sp, 1);
        Tensor v = SplitQkvTemporal(backend, qkv, tt, sp, 2);
        qkv.Dispose();
        RmsNormHeads(backend, q, _imgQNorm!);
        RmsNormHeads(backend, k, _imgKNorm!);
        ApplyRopeBatched(backend, q, cos, sin, tt, gh, gw);
        ApplyRopeBatched(backend, k, cos, sin, tt, gh, gw);
        cos.Dispose(); sin.Dispose();

        Tensor attn = new Tensor(new TensorShape([(long)sp, _heads, tt, _headDim]), DType.F32);
        // cuDNN fused flash: mask-null, D=64, batched over sp spatial positions × tiny frame-seq — the same
        // shape that made Oasis's materialized SDPA catastrophic. allowF16 (bounded/RMS-normed q/k).
        backend.ScaledDotProductAttention(attn, q, k, v, null, 1.0f / MathF.Sqrt(_headDim), allowF16: true);
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor flat = MergeTemporal(backend, attn, tt, sp);
        attn.Dispose();
        Tensor projected = new Tensor(new TensorShape(tt * sp, _imgDim), DType.F32);
        backend.Linear(projected, flat, _projMouseW!, _projMouseB);
        flat.Dispose();
        backend.Add(hidden, hidden, projected);
        projected.Dispose();
    }

    private void KeyboardStream(IBackend backend, Tensor hidden, int tt, int gh, int gw, ReadOnlySpan<int> frameIndices,
        int memoryFrames, Tensor keyboard)
    {
        int sp = gh * gw;
        int rawFrames = (int)keyboard.Shape[0];

        // Embed raw keyboard frames (6 → 128, SiLU between), then window-group per latent frame → [tt, 128·12].
        Tensor e1 = new Tensor(new TensorShape(rawFrames, _hiddenSize), DType.F32);
        backend.Linear(e1, keyboard, _kbdEmbed1W!, _kbdEmbed1B);
        Tensor e1a = new Tensor(e1.Shape, DType.F32); backend.Silu(e1a, e1); e1.Dispose();
        Tensor embedded = new Tensor(new TensorShape(rawFrames, _hiddenSize), DType.F32);
        backend.Linear(embedded, e1a, _kbdEmbed2W!, _kbdEmbed2B);
        e1a.Dispose();
        float[] grouped = BuildWindows(embedded, _hiddenSize, tt, memoryFrames);   // [tt, winLen·128]
        embedded.Dispose();

        int kvIn = _hiddenSize * WindowLength;
        Tensor groupedT = new Tensor(new TensorShape(tt, kvIn), DType.F32);
        fixed (float* gp = grouped)
            Buffer.MemoryCopy(gp, (float*)groupedT.DataPointer, (long)tt * kvIn * 4, (long)tt * kvIn * 4);
        Tensor kv = new Tensor(new TensorShape(tt, 2 * _streamDim), DType.F32);
        backend.Linear(kv, groupedT, _kbdKvW!, _kbdKvB);
        groupedT.Dispose();

        Tensor qFlat = new Tensor(new TensorShape(tt * sp, _streamDim), DType.F32);
        backend.Linear(qFlat, hidden, _kbdQW!, _kbdQB);

        // Q per spatial position over frames; K/V broadcast across positions. RoPE rotates the temporal axis only
        // (grid 1×1 — keys carry no spatial coordinate because they are shared by every position).
        (Tensor cosT, Tensor sinT) = _rope.BuildCosSin(frameIndices, 1, 1);
        Tensor q = SplitQkvTemporal(backend, qFlat, tt, sp, 0, stride: 1);
        qFlat.Dispose();
        RmsNormHeads(backend, q, _keyQNorm!);
        ApplyRopeBatched(backend, q, cosT, sinT, tt, 1, 1, broadcastSpatial: true);

        // K/V are shared by every spatial position → expand kv[tt, 2·streamDim] directly to [sp, heads, tt, headDim]
        // (device). RMS-norm + rope on the broadcast K are per-(head,frame) so all sp rows stay identical → equals the
        // old norm-then-broadcast, without the host K/V build loop or BroadcastBatch.
        Tensor k = new Tensor(new TensorShape([(long)sp, _heads, tt, _headDim]), DType.F32);
        Tensor v = new Tensor(new TensorShape([(long)sp, _heads, tt, _headDim]), DType.F32);
        backend.Mg3KvExpand(k, v, kv, sp, _heads, tt, _headDim);
        kv.Dispose();
        RmsNormHeads(backend, k, _keyKNorm!);
        ApplyRopeBatched(backend, k, cosT, sinT, tt, 1, 1, broadcastSpatial: true);
        cosT.Dispose(); sinT.Dispose();

        Tensor attn = new Tensor(new TensorShape([(long)sp, _heads, tt, _headDim]), DType.F32);
        backend.ScaledDotProductAttention(attn, q, k, v, null, 1.0f / MathF.Sqrt(_headDim), allowF16: true);
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor flat = MergeTemporal(backend, attn, tt, sp);
        attn.Dispose();
        Tensor projected = new Tensor(new TensorShape(tt * sp, _imgDim), DType.F32);
        backend.Linear(projected, flat, _projKbdW!, _projKbdB);
        flat.Dispose();
        backend.Add(hidden, hidden, projected);
        projected.Dispose();
    }

    /// <summary>Token rows <c>[(f·sp + s), stride·streamDim]</c> → batched temporal sequences <c>[sp, heads, tt, headDim]</c>,
    /// taking QKV slot <paramref name="part"/>. Device-resident via <see cref="IBackend.Mg3SplitQkvTemporal"/> (the host
    /// pointer-loop was a dominant per-block cost).</summary>
    private Tensor SplitQkvTemporal(IBackend backend, Tensor qkv, int tt, int sp, int part, int stride = 3)
    {
        Tensor o = new Tensor(new TensorShape([(long)sp, _heads, tt, _headDim]), DType.F32);
        backend.Mg3SplitQkvTemporal(o, qkv, tt, sp, _heads, _headDim, part, stride);
        return o;
    }

    /// <summary>Back from <c>[sp, heads, tt, headDim]</c> to token rows <c>[tt·sp, streamDim]</c>, device-resident.</summary>
    private Tensor MergeTemporal(IBackend backend, Tensor attn, int tt, int sp)
    {
        Tensor o = new Tensor(new TensorShape(tt * sp, _streamDim), DType.F32);
        backend.Mg3MergeTemporal(o, attn, tt, sp, _heads, _headDim);
        return o;
    }

    /// <summary>Per-head RMSNorm over the trailing head dim of <c>[B, heads, seq, headDim]</c> — device
    /// <see cref="IBackend.RmsNorm"/> (last-dim RMS × weight), in place.</summary>
    private void RmsNormHeads(IBackend backend, Tensor x, Tensor weight)
        => backend.RmsNorm(x, x, weight, _eps);

    /// <summary>Rotates <c>[sp, heads, tt, headDim]</c> with cos/sin built over the (frames, gh, gw) grid: the sequence at
    /// spatial position s uses grid row <c>f·gh·gw + s</c> (or <c>f</c> when <paramref name="broadcastSpatial"/>).
    /// Device-resident via <see cref="IBackend.Mg3RopeBatched"/>.</summary>
    private void ApplyRopeBatched(IBackend backend, Tensor x, Tensor cos, Tensor sin, int tt, int gh, int gw, bool broadcastSpatial = false)
    {
        int sp = (int)x.Shape[0];
        backend.Mg3RopeBatched(x, cos, sin, sp, _heads, tt, _headDim, gh, gw, broadcastSpatial);
    }

    /// <summary>Affine LayerNorm over the trailing <paramref name="dim"/> of <paramref name="x"/> <c>[rows, dim]</c>,
    /// in place, device-resident: no-affine LN then a row-broadcast <c>·weight + bias</c> (weight/bias are <c>[dim]</c>,
    /// shared across all rows — <see cref="IBackend.AffineBroadcastLastDim"/> at batch 1).</summary>
    private void LayerNormRows(IBackend backend, Tensor x, Tensor weight, Tensor? bias, int rows, int dim)
    {
        backend.LayerNormNoAffine(x, x, _eps);
        backend.AffineBroadcastLastDim(x, x, weight, bias);
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string k) { Tensor t = w[k]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }
    private static Tensor? LoadF32Opt(IReadOnlyDictionary<string, Tensor> w, string k) => w.TryGetValue(k, out Tensor? t) ? (t.DType == DType.F32 ? t : t.CastTo(DType.F32)) : null;
}
