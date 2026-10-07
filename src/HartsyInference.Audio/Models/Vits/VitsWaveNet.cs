using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Vits;

/// <summary>VITS WaveNet residual stack (the <c>WN</c> module shared by the flow's coupling layers and the posterior encoder): N dilated gated layers — <c>in_layer</c> (dilated Conv1d → 2·hidden) → fused tanh·sigmoid gate → <c>res_skip_layer</c>; residual feeds the next layer, skips accumulate; optional speaker conditioning (<c>g</c>) is projected once and added per layer.</summary>
public sealed unsafe class VitsWaveNet
{
    private readonly int _hidden, _layers, _kernel, _dilationRate;
    private readonly Tensor?[] _inW, _inB, _rsW, _rsB;
    private Tensor? _condW, _condB;     // cond_layer (multispeaker g), optional
    private Tensor?[]? _condWLayers, _condBLayers;   // per-layer row views (persistent, so a backend's weight cache keeps them resident)

    public VitsWaveNet(int hidden, int kernel, int dilationRate, int layers)
    {
        _hidden = hidden; _kernel = kernel; _dilationRate = dilationRate; _layers = layers;
        _inW = new Tensor?[layers]; _inB = new Tensor?[layers];
        _rsW = new Tensor?[layers]; _rsB = new Tensor?[layers];
    }

    /// <param name="convKeySuffix">Extra path segment between each conv's own name and its
    /// <c>weight</c>/<c>weight_g</c>/<c>weight_v</c>/<c>bias</c> leaf — VITS's own checkpoints need none (the
    /// default, empty string); IndexTTS-2's S2Mel WaveNet wraps every conv in an <c>SConv1d</c> "streamable
    /// conv" module, nesting the real weights one level deeper (<c>in_layers.0.conv.conv.weight_g</c> instead
    /// of <c>in_layers.0.weight_g</c>), so it passes <c>".conv.conv"</c>. Confirmed from the real
    /// <c>s2mel.pth</c>'s own key names — this nesting is NOT a VITS convention and would not be guessable
    /// from the Python source's naming alone.</param>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix, string convKeySuffix = "")
    {
        for (int i = 0; i < _layers; i++)
        {
            _inW[i] = VitsWeights.Conv(w, $"{prefix}.in_layers.{i}{convKeySuffix}");
            _inB[i] = VitsWeights.Bias(w, $"{prefix}.in_layers.{i}{convKeySuffix}");
            _rsW[i] = VitsWeights.Conv(w, $"{prefix}.res_skip_layers.{i}{convKeySuffix}");
            _rsB[i] = VitsWeights.Bias(w, $"{prefix}.res_skip_layers.{i}{convKeySuffix}");
        }
        if (w.ContainsKey($"{prefix}.cond_layer{convKeySuffix}.weight") || w.ContainsKey($"{prefix}.cond_layer{convKeySuffix}.weight_g"))
        {
            _condW = VitsWeights.Conv(w, $"{prefix}.cond_layer{convKeySuffix}");
            _condB = VitsWeights.Bias(w, $"{prefix}.cond_layer{convKeySuffix}");
            _condWLayers = new Tensor?[_layers];
            _condBLayers = new Tensor?[_layers];
            for (int i = 0; i < _layers; i++)
            {
                _condWLayers[i] = _condW!.SliceRows((long)i * 2 * _hidden, 2 * _hidden);
                _condBLayers[i] = _condB!.SliceRows((long)i * 2 * _hidden, 2 * _hidden);
            }
        }
    }

    /// <summary>Runs the stack over <paramref name="x"/> <c>[1, hidden, T]</c>; <paramref name="g"/> is the optional speaker embedding <c>[1, gin, 1]</c> (multispeaker), projected per layer into the gate. Everything is a backend op (conditioning add, split, gate, residual and skip accumulation) so no activation visits the host between layers.</summary>
    /// <returns>The accumulated skip sum.</returns>
    public Tensor Forward(IBackend backend, Tensor x, int t, Tensor? g = null)
    {
        int h = _hidden;
        Tensor? skipSum = null;
        Tensor? cur = null;   // null until the first layer produces a new residual; x itself is never mutated or disposed

        for (int i = 0; i < _layers; i++)
        {
            Tensor input = cur ?? x;
            int dilation = Pow(_dilationRate, i);
            int pad = dilation * (_kernel - 1) / 2;
            Tensor xIn = new(new TensorShape(1, 2 * h, t), DType.F32);
            backend.Conv1d(xIn, input, _inW[i]!, _inB[i], 1, pad, pad, dilation, 1);

            // Per-layer speaker-conditioning slice (broadcast over time): this layer's rows of cond_layer applied to g.
            if (g is not null && _condWLayers is not null)
            {
                using Tensor gLayer = new(new TensorShape(1, 2 * h, 1), DType.F32);
                backend.Conv1d(gLayer, g, _condWLayers[i]!, _condBLayers![i], 1, 0, 0, 1, 1);
                using Tensor gBias = gLayer.Reshape(new TensorShape(1, 2 * h));
                backend.BroadcastAdd(xIn, gBias, 2 * h, t);
            }

            // fused tanh(first h) * sigmoid(second h) → acts [1, h, t].
            Tensor first = new(new TensorShape(1, h, t), DType.F32);
            Tensor second = new(new TensorShape(1, h, t), DType.F32);
            backend.Split([first, second], xIn, 1);
            xIn.Dispose();
            backend.Tanh(first, first);
            backend.Sigmoid(second, second);
            Tensor acts = new(new TensorShape(1, h, t), DType.F32);
            backend.Mul(acts, first, second);
            first.Dispose();
            second.Dispose();

            int rsCh = i < _layers - 1 ? 2 * h : h;
            Tensor rs = new(new TensorShape(1, rsCh, t), DType.F32);
            backend.Conv1d(rs, acts, _rsW[i]!, _rsB[i], 1, 0, 0, 1, 1);
            acts.Dispose();

            if (i < _layers - 1)
            {
                // residual = (cur + rs[:h]); skip = rs[h:].
                Tensor residual = new(new TensorShape(1, h, t), DType.F32);
                Tensor skip = new(new TensorShape(1, h, t), DType.F32);
                backend.Split([residual, skip], rs, 1);
                rs.Dispose();

                Tensor newCur = new(new TensorShape(1, h, t), DType.F32);
                backend.Add(newCur, input, residual);
                residual.Dispose();
                cur?.Dispose();
                cur = newCur;

                skipSum = AccumulateSkip(backend, skipSum, skip);
            }
            else
            {
                skipSum = AccumulateSkip(backend, skipSum, rs);
            }
        }
        cur?.Dispose();
        return skipSum!;
    }

    /// <summary><c>sum + skip</c>; the first skip becomes the running sum itself (the old zero-initialised accumulator, minus the add).</summary>
    private static Tensor AccumulateSkip(IBackend backend, Tensor? sum, Tensor skip)
    {
        if (sum is null) return skip;
        Tensor next = new(sum.Shape, DType.F32);
        backend.Add(next, sum, skip);
        sum.Dispose();
        skip.Dispose();
        return next;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[][] groups = [_inW, _inB, _rsW, _rsB];
        foreach (Tensor?[] g in groups) foreach (Tensor? t in g) if (t is not null) yield return t;
        if (_condW is not null) yield return _condW;
        if (_condB is not null) yield return _condB;
    }

    private static int Pow(int b, int e) { int r = 1; for (int i = 0; i < e; i++) r *= b; return r; }
}
