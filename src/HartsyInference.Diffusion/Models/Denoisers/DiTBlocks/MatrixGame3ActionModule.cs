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
            _mouseLnW = LoadF32(w, $"{prefix}.mouse_mlp.3.weight"); w.TryGetValue($"{prefix}.mouse_mlp.3.bias", out _mouseLnB);
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
    /// covers raw frames <c>[i·ratio − ratio·(window−1), i·ratio + ratio)</c>, left-padded by repeating row 0.
    /// Memory frames produce zero windows.</summary>
    private float[] BuildWindows(Tensor raw, int dim, int tt, int memoryFrames)
    {
        int winLen = WindowLength;
        int rawFrames = (int)raw.Shape[0];
        float* rp = (float*)raw.DataPointer;
        float[] result = new float[tt * winLen * dim];
        for (int f = memoryFrames; f < tt; f++)
        {
            int local = f - memoryFrames;
            int start = local * _ratio - _ratio * (_window - 1);
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
        float* hp = (float*)hidden.DataPointer;

        // Per-token MLP input: [token features (dim) | grouped mouse window (winFloats)] — sequence axis is the frame.
        Tensor mlpIn = new Tensor(new TensorShape(tt * sp, _imgDim + winFloats), DType.F32);
        float* mip = (float*)mlpIn.DataPointer;
        for (int f = 0; f < tt; f++)
            for (int s = 0; s < sp; s++)
            {
                long token = (long)f * sp + s;
                Buffer.MemoryCopy(hp + token * _imgDim, mip + token * (_imgDim + winFloats), (long)_imgDim * 4, (long)_imgDim * 4);
                for (int wf = 0; wf < winFloats; wf++)
                    mip[token * (_imgDim + winFloats) + _imgDim + wf] = mouseWin[f * winFloats + wf];
            }

        Tensor h1 = new Tensor(new TensorShape(tt * sp, _streamDim), DType.F32);
        backend.Linear(h1, mlpIn, _mouseMlp1W!, _mouseMlp1B);
        mlpIn.Dispose();
        Tensor act = new Tensor(h1.Shape, DType.F32); backend.Gelu(act, h1); h1.Dispose();
        Tensor h2 = new Tensor(new TensorShape(tt * sp, _streamDim), DType.F32);
        backend.Linear(h2, act, _mouseMlp2W!, _mouseMlp2B);
        act.Dispose();
        LayerNormRows(h2, _mouseLnW!, _mouseLnB, tt * sp, _streamDim);

        Tensor qkv = new Tensor(new TensorShape(tt * sp, 3 * _streamDim), DType.F32);
        backend.Linear(qkv, h2, _tQkvW!, _tQkvB);
        h2.Dispose();

        // Temporal self-attention per spatial position, batched as [sp, heads, tt, headDim].
        (Tensor cos, Tensor sin) = _rope.BuildCosSin(frameIndices, gh, gw);
        Tensor q = SplitQkvTemporal(qkv, tt, sp, 0);
        Tensor k = SplitQkvTemporal(qkv, tt, sp, 1);
        Tensor v = SplitQkvTemporal(qkv, tt, sp, 2);
        qkv.Dispose();
        RmsNormHeads(q, _imgQNorm!);
        RmsNormHeads(k, _imgKNorm!);
        ApplyRopeBatched(q, cos, sin, tt, gh, gw);
        ApplyRopeBatched(k, cos, sin, tt, gh, gw);
        cos.Dispose(); sin.Dispose();

        Tensor attn = new Tensor(new TensorShape([(long)sp, _heads, tt, _headDim]), DType.F32);
        backend.ScaledDotProductAttention(attn, q, k, v, null, 1.0f / MathF.Sqrt(_headDim));
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor flat = MergeTemporal(attn, tt, sp);
        attn.Dispose();
        Tensor projected = new Tensor(new TensorShape(tt * sp, _imgDim), DType.F32);
        backend.Linear(projected, flat, _projMouseW!, _projMouseB);
        flat.Dispose();
        AddInPlace(hidden, projected);
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
        Tensor q = SplitQkvTemporal(qFlat, tt, sp, 0, stride: 1);
        qFlat.Dispose();
        RmsNormHeads(q, _keyQNorm!);
        ApplyRopeBatched(q, cosT, sinT, tt, 1, 1, broadcastSpatial: true);

        Tensor k = new Tensor(new TensorShape([1L, _heads, tt, _headDim]), DType.F32);
        Tensor v = new Tensor(new TensorShape([1L, _heads, tt, _headDim]), DType.F32);
        float* kvp = (float*)kv.DataPointer;
        float* kp = (float*)k.DataPointer;
        float* vp = (float*)v.DataPointer;
        for (int f = 0; f < tt; f++)
            for (int h = 0; h < _heads; h++)
                for (int d = 0; d < _headDim; d++)
                {
                    kp[((long)h * tt + f) * _headDim + d] = kvp[(long)f * 2 * _streamDim + h * _headDim + d];
                    vp[((long)h * tt + f) * _headDim + d] = kvp[(long)f * 2 * _streamDim + _streamDim + h * _headDim + d];
                }
        kv.Dispose();
        RmsNormHeads(k, _keyKNorm!);
        ApplyRopeBatched(k, cosT, sinT, tt, 1, 1, broadcastSpatial: true);
        cosT.Dispose(); sinT.Dispose();

        // Broadcast K/V to every spatial position.
        Tensor kB = BroadcastBatch(k, sp, tt); k.Dispose();
        Tensor vB = BroadcastBatch(v, sp, tt); v.Dispose();
        Tensor attn = new Tensor(new TensorShape([(long)sp, _heads, tt, _headDim]), DType.F32);
        backend.ScaledDotProductAttention(attn, q, kB, vB, null, 1.0f / MathF.Sqrt(_headDim));
        q.Dispose(); kB.Dispose(); vB.Dispose();

        Tensor flat = MergeTemporal(attn, tt, sp);
        attn.Dispose();
        Tensor projected = new Tensor(new TensorShape(tt * sp, _imgDim), DType.F32);
        backend.Linear(projected, flat, _projKbdW!, _projKbdB);
        flat.Dispose();
        AddInPlace(hidden, projected);
        projected.Dispose();
    }

    /// <summary>Token rows <c>[(f·sp + s), stride·streamDim]</c> → batched temporal sequences <c>[sp, heads, tt, headDim]</c>, taking QKV slot <paramref name="part"/>.</summary>
    private Tensor SplitQkvTemporal(Tensor qkv, int tt, int sp, int part, int stride = 3)
    {
        Tensor o = new Tensor(new TensorShape([(long)sp, _heads, tt, _headDim]), DType.F32);
        float* src = (float*)qkv.DataPointer;
        float* dst = (float*)o.DataPointer;
        int rowDim = stride * _streamDim;
        for (int f = 0; f < tt; f++)
            for (int s = 0; s < sp; s++)
            {
                long token = (long)f * sp + s;
                for (int h = 0; h < _heads; h++)
                    Buffer.MemoryCopy(
                        src + token * rowDim + part * _streamDim + h * _headDim,
                        dst + (((long)s * _heads + h) * tt + f) * _headDim,
                        (long)_headDim * 4, (long)_headDim * 4);
            }
        return o;
    }

    /// <summary>Back from <c>[sp, heads, tt, headDim]</c> to token rows <c>[tt·sp, streamDim]</c>.</summary>
    private Tensor MergeTemporal(Tensor attn, int tt, int sp)
    {
        Tensor o = new Tensor(new TensorShape(tt * sp, _streamDim), DType.F32);
        float* src = (float*)attn.DataPointer;
        float* dst = (float*)o.DataPointer;
        for (int s = 0; s < sp; s++)
            for (int h = 0; h < _heads; h++)
                for (int f = 0; f < tt; f++)
                    Buffer.MemoryCopy(
                        src + (((long)s * _heads + h) * tt + f) * _headDim,
                        dst + ((long)f * sp + s) * _streamDim + h * _headDim,
                        (long)_headDim * 4, (long)_headDim * 4);
        return o;
    }

    /// <summary>Per-head RMSNorm over the trailing head dim of <c>[B, heads, seq, headDim]</c>.</summary>
    private void RmsNormHeads(Tensor x, Tensor weight)
    {
        long rows = x.Shape.ElementCount / _headDim;
        float* xp = (float*)x.DataPointer;
        float* wp = (float*)weight.DataPointer;
        for (long r = 0; r < rows; r++)
        {
            long off = r * _headDim;
            double sum = 0;
            for (int d = 0; d < _headDim; d++) { float v = xp[off + d]; sum += (double)v * v; }
            float inv = 1f / MathF.Sqrt((float)(sum / _headDim) + _eps);
            for (int d = 0; d < _headDim; d++) xp[off + d] = xp[off + d] * inv * wp[d];
        }
    }

    /// <summary>Rotates <c>[sp(or 1), heads, tt, headDim]</c> with cos/sin built over the (frames, gh, gw) grid: the
    /// sequence at spatial position s uses the grid row (f, h(s), w(s)).</summary>
    private void ApplyRopeBatched(Tensor x, Tensor cos, Tensor sin, int tt, int gh, int gw, bool broadcastSpatial = false)
    {
        int sp = (int)x.Shape[0];
        float* xp = (float*)x.DataPointer;
        float* cp = (float*)cos.DataPointer;
        float* spn = (float*)sin.DataPointer;
        int pairs = _headDim / 2;
        for (int s = 0; s < sp; s++)
            for (int h = 0; h < _heads; h++)
                for (int f = 0; f < tt; f++)
                {
                    long gridRow = broadcastSpatial ? f : (long)f * gh * gw + s;
                    long cOff = gridRow * _headDim;
                    long xOff = (((long)s * _heads + h) * tt + f) * _headDim;
                    for (int i = 0; i < pairs; i++)
                    {
                        int i0 = 2 * i;
                        float re = xp[xOff + i0], im = xp[xOff + i0 + 1];
                        float c = cp[cOff + i0], sn = spn[cOff + i0];
                        xp[xOff + i0] = re * c - im * sn;
                        xp[xOff + i0 + 1] = re * sn + im * c;
                    }
                }
    }

    private Tensor BroadcastBatch(Tensor x, int batch, int tt)
    {
        Tensor o = new Tensor(new TensorShape([(long)batch, _heads, tt, _headDim]), DType.F32);
        long per = (long)_heads * tt * _headDim;
        float* src = (float*)x.DataPointer;
        float* dst = (float*)o.DataPointer;
        for (int b = 0; b < batch; b++)
            Buffer.MemoryCopy(src, dst + b * per, per * 4, per * 4);
        return o;
    }

    private void LayerNormRows(Tensor x, Tensor weight, Tensor? bias, int rows, int dim)
    {
        float* xp = (float*)x.DataPointer;
        float* wp = (float*)weight.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;
        for (int r = 0; r < rows; r++)
        {
            long off = (long)r * dim;
            double mean = 0;
            for (int d = 0; d < dim; d++) mean += xp[off + d];
            mean /= dim;
            double var = 0;
            for (int d = 0; d < dim; d++) { double dd = xp[off + d] - mean; var += dd * dd; }
            float inv = 1f / MathF.Sqrt((float)(var / dim) + _eps);
            for (int d = 0; d < dim; d++)
            {
                float n = (float)((xp[off + d] - mean) * inv);
                xp[off + d] = n * wp[d] + (bp is null ? 0f : bp[d]);
            }
        }
    }

    private static void AddInPlace(Tensor target, Tensor value)
    {
        long n = target.Shape.ElementCount;
        float* tp = (float*)target.DataPointer;
        float* vp = (float*)value.DataPointer;
        for (long i = 0; i < n; i++) tp[i] += vp[i];
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string k) { Tensor t = w[k]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }
}
