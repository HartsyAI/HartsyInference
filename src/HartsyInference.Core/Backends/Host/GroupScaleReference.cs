using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Gating;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>The executable form of the group-scale contract (<c>docs/BACKEND_OPS.md</c>): group <c>g</c> of row <c>r</c> times
/// <c>f(gates[offset + r · stride + g])</c>, written or accumulated (a zero gate leaves an accumulated row untouched). The
/// <see cref="IBackend"/> default and the oracle every kernel is checked against.</summary>
public static unsafe class GroupScaleReference
{
    /// <summary>The factor a stored gate becomes under <paramref name="function"/>.</summary>
    public static float Factor(float gate, GateFunction function)
        => function == GateFunction.TwoSigmoid ? 2f * GluReference.Sigmoid(gate) : gate;

    public static void Apply(Tensor output, Tensor input, Tensor gates, in GroupScaleSpec spec)
    {
        GroupScaleGeometry g = GroupScaleContract.Validate(output, input, gates, spec);
        Func<long, float> load = GluReference.Loader(input);
        Func<long, float> loadOut = GluReference.Loader(output);
        Func<long, float> loadGate = GluReference.Loader(gates);
        Action<long, float> store = GluReference.Storer(output);
        for (long r = 0; r < g.Rows; r++)
        {
            for (int group = 0; group < g.Groups; group++)
            {
                float factor = Factor(loadGate(spec.GateOffset + r * spec.GateStride + group), spec.Function);
                if (spec.Accumulate && factor == 0f) continue;
                long at = r * g.Width + (long)group * g.GroupWidth;
                for (int d = 0; d < g.GroupWidth; d++)
                {
                    float v = factor * load(at + d);
                    store(at + d, spec.Accumulate ? loadOut(at + d) + v : v);
                }
            }
        }
    }
}
