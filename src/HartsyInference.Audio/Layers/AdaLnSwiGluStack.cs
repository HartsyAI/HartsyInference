using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Layers;

/// <summary>The denoiser body of a continuous-latent speech head: N layers of
/// <c>shift, scale, gate = ada(c)</c>, <c>x += gate * SwiGLU(RMSNorm(x) * w * (1 + scale) + shift)</c>, then a final layer
/// <c>proj(RMSNorm(x) * (1 + scale) + shift)</c> with a weightless RMSNorm. None of the linears carries a bias.</summary>
/// <remarks>The conditioning enters as <c>c</c> already through SiLU, once per call: both heads built on this stack apply
/// <c>ada = Linear(silu(c))</c> in every layer.</remarks>
internal sealed unsafe class AdaLnSwiGluStack : IDisposable
{
    private readonly int _hidden;
    private readonly int _ffn;
    private readonly float _eps;
    private readonly int _outDim;
    private readonly Layer[] _layers;
    private readonly Tensor _finalAda;
    private readonly Tensor _finalProj;
    private readonly Tensor _ones;

    /// <summary>Weights of one layer: the norm weight (<c>[H]</c>), <c>ada</c> (<c>[3H, H]</c>), and the SwiGLU's
    /// <c>gate</c>, <c>up</c> (<c>[F, H]</c>) and <c>down</c> (<c>[H, F]</c>).</summary>
    internal readonly record struct Layer(Tensor Norm, Tensor Ada, Tensor Gate, Tensor Up, Tensor Down);

    /// <summary>The linears are used as stored (BF16 stays BF16); the norm weights must be F32. The tensors stay owned by the caller.</summary>
    internal AdaLnSwiGluStack(int hidden, int ffn, float eps, int outDim, IReadOnlyList<Layer> layers, Tensor finalAda, Tensor finalProj)
    {
        ArgumentNullException.ThrowIfNull(layers);
        ArgumentNullException.ThrowIfNull(finalAda);
        ArgumentNullException.ThrowIfNull(finalProj);
        _hidden = hidden;
        _ffn = ffn;
        _eps = eps;
        _outDim = outDim;
        _layers = [.. layers];
        _finalAda = Check(finalAda, 2 * hidden, hidden, nameof(finalAda));
        _finalProj = Check(finalProj, outDim, hidden, nameof(finalProj));
        for (int i = 0; i < _layers.Length; i++)
        {
            Layer l = _layers[i];
            if (l.Norm.DType != DType.F32 || l.Norm.ElementCount != hidden)
            {
                throw new ArgumentException($"Layer {i} norm must be F32 [{hidden}], got {l.Norm.DType} {l.Norm.Shape}.", nameof(layers));
            }
            Check(l.Ada, 3 * hidden, hidden, nameof(layers));
            Check(l.Gate, ffn, hidden, nameof(layers));
            Check(l.Up, ffn, hidden, nameof(layers));
            Check(l.Down, hidden, ffn, nameof(layers));
        }
        _ones = new Tensor(new TensorShape(hidden), DType.F32);
        new Span<float>((void*)_ones.DataPointer, hidden).Fill(1f);
    }

    /// <summary>Runs the stack on <paramref name="x"/> (<c>[1, N, H]</c>) with the SiLU'd conditioning
    /// <paramref name="activeCondition"/> (<c>[1, N, H]</c>); neither input is consumed. Returns <c>[1, N, outDim]</c>, owned by the caller.</summary>
    internal Tensor Forward(IBackend backend, Tensor x, Tensor activeCondition)
    {
        int n = (int)x.Shape[1];
        Tensor current = LayerForward(backend, x, activeCondition, _layers[0], n);
        for (int i = 1; i < _layers.Length; i++)
        {
            Tensor next = LayerForward(backend, current, activeCondition, _layers[i], n);
            current.Dispose();
            current = next;
        }
        Tensor result = FinalForward(backend, current, activeCondition, n);
        current.Dispose();
        return result;
    }

    internal IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Layer l in _layers)
        {
            yield return l.Norm;
            yield return l.Ada;
            yield return l.Gate;
            yield return l.Up;
            yield return l.Down;
        }
        yield return _finalAda;
        yield return _finalProj;
        yield return _ones;
    }

    public void Dispose() => _ones.Dispose();

    private static Tensor Check(Tensor weight, int rows, int cols, string name)
    {
        if (weight.Shape.Rank != 2 || weight.Shape[0] != rows || weight.Shape[1] != cols)
        {
            throw new ArgumentException($"Weight must be [{rows}, {cols}], got {weight.Shape}.", name);
        }
        return weight;
    }

    private Tensor LayerForward(IBackend backend, Tensor x, Tensor activeCondition, Layer l, int n)
    {
        Tensor mod = Project(backend, activeCondition, l.Ada, n, 3 * _hidden);
        Tensor shift = Slice(backend, mod, n, 0);
        Tensor scale = Slice(backend, mod, n, _hidden);
        Tensor gate = Slice(backend, mod, n, 2 * _hidden);
        mod.Dispose();

        Tensor normed = new(x.Shape, DType.F32);
        backend.RmsNorm(normed, x, l.Norm, _eps);
        Tensor modulated = Modulate(backend, normed, shift, scale);
        normed.Dispose();
        shift.Dispose();
        scale.Dispose();

        Tensor swiGlu = SwiGlu(backend, modulated, l, n);
        modulated.Dispose();
        Tensor gated = new(swiGlu.Shape, DType.F32);
        backend.Mul(gated, gate, swiGlu);
        gate.Dispose();
        swiGlu.Dispose();
        Tensor result = new(x.Shape, DType.F32);
        backend.Add(result, x, gated);
        gated.Dispose();
        return result;
    }

    private Tensor FinalForward(IBackend backend, Tensor x, Tensor activeCondition, int n)
    {
        Tensor mod = Project(backend, activeCondition, _finalAda, n, 2 * _hidden);
        Tensor shift = Slice(backend, mod, n, 0);
        Tensor scale = Slice(backend, mod, n, _hidden);
        mod.Dispose();

        Tensor normed = new(x.Shape, DType.F32);
        backend.RmsNorm(normed, x, _ones, _eps);
        Tensor modulated = Modulate(backend, normed, shift, scale);
        normed.Dispose();
        shift.Dispose();
        scale.Dispose();
        Tensor result = Project(backend, modulated, _finalProj, n, _outDim);
        modulated.Dispose();
        return result;
    }

    private Tensor SwiGlu(IBackend backend, Tensor x, Layer l, int n)
    {
        Tensor gate = Project(backend, x, l.Gate, n, _ffn);
        Tensor up = Project(backend, x, l.Up, n, _ffn);
        Tensor gateActive = new(gate.Shape, DType.F32);
        backend.Silu(gateActive, gate);
        gate.Dispose();
        Tensor mixed = new(gateActive.Shape, DType.F32);
        backend.Mul(mixed, gateActive, up);
        gateActive.Dispose();
        up.Dispose();
        Tensor result = Project(backend, mixed, l.Down, n, _hidden);
        mixed.Dispose();
        return result;
    }

    private static Tensor Project(IBackend backend, Tensor input, Tensor weight, int n, int outDim)
    {
        Tensor output = new(new TensorShape(1, n, outDim), DType.F32);
        backend.Linear(output, input, weight, null);
        return output;
    }

    private Tensor Slice(IBackend backend, Tensor source, int n, int offset)
    {
        Tensor result = new(new TensorShape(1, n, _hidden), DType.F32);
        backend.SliceLastDim(result, source, offset);
        return result;
    }

    private static Tensor Modulate(IBackend backend, Tensor x, Tensor shift, Tensor scale)
    {
        Tensor onePlusScale = new(scale.Shape, DType.F32);
        backend.AddScalar(onePlusScale, scale, 1f);
        Tensor scaled = new(x.Shape, DType.F32);
        backend.Mul(scaled, x, onePlusScale);
        onePlusScale.Dispose();
        Tensor result = new(x.Shape, DType.F32);
        backend.Add(result, scaled, shift);
        scaled.Dispose();
        return result;
    }
}
