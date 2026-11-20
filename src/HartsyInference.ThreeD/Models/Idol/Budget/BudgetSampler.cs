using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Sampling;

namespace HartsyInference.ThreeD.Models.Idol.Budget;

/// <summary>Sampling Budget3D at a token budget (upstream <c>budget3d/sampling/sampler.py</c> and <c>turbo.py</c>): Euler integration of the rectified flow on Idol's shifted time grid, with the layout built from the
/// noise at the first step and rebuilt from the running clean estimate <c>z0 = z − t·v</c>. The base model rebuilds it at steps 0 and 5 and then freezes it (<c>relayoutEvery = 5</c>, <c>relayoutUntil = 5</c>); the
/// turbo model rebuilds it at every step (<c>relayoutEvery = 1</c>, <c>relayoutUntil = −1</c>). The loop is explicit rather than on <see cref="RectifiedFlowSampler"/> because the layout needs the guided velocity of
/// each step; the time grid is the shared one.</summary>
public static class BudgetSampler
{
    /// <summary>The base model's defaults: 50 steps, guidance 3, time shift 3, a layout refreshed at steps 0 and 5.</summary>
    public const int DefaultSteps = IdolSampler.DefaultSteps;

    /// <summary>Layout refresh cadence of the base model.</summary>
    public const int BaseRelayoutEvery = 5;

    /// <summary>The last step of the base model that may refresh the layout.</summary>
    public const int BaseRelayoutUntil = 5;

    /// <summary>Denoises <paramref name="noiseLatent"/> <c>[1, 8192, 16]</c> and <paramref name="noiseCamera"/> <c>[1, 1, 5]</c> (neither modified) at <paramref name="tokens"/> tokens. <paramref name="positive"/> and
    /// <paramref name="negative"/> are the image streams of the picture and of zeros (the negative is unused at guidance 0); <paramref name="contextAt"/> replaces them for a model whose stream depends on the time
    /// (the turbo's), and <paramref name="passScope"/> opens a scope around each forward pass given its <c>1000·t</c> (the turbo's phase expert). Returns the final layout too. The caller owns both tensors.</summary>
    public static (Tensor Latent, Tensor Camera, BudgetLayout Layout) Sample(IBackend backend, BudgetDenoiser model, Tensor positive, Tensor? negative, Tensor noiseLatent, Tensor noiseCamera, int tokens,
        int steps, float guidance, double rescaleT, int relayoutEvery, int relayoutUntil, Action<int>? onStep = null, Func<float, IDisposable>? passScope = null, Func<float, Tensor>? contextAt = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(model);
        if (guidance < 0f || !float.IsFinite(guidance))
            throw new ArgumentOutOfRangeException(nameof(guidance), guidance, "Guidance must be finite and not negative.");
        if (guidance > 0f && negative is null && contextAt is null)
            throw new ArgumentException("Guidance needs the image stream of the blank picture.", nameof(negative));
        if (relayoutEvery < 1)
            throw new ArgumentOutOfRangeException(nameof(relayoutEvery), relayoutEvery, "The layout is refreshed every step or less often.");
        (int[] toOctree, _) = BudgetOctree.Permutation(IdolWeights.EmbeddedPositions());
        int lc = (int)noiseLatent.Shape[2], cc = (int)noiseCamera.Shape[2];
        float[] z = noiseLatent.AsReadOnlySpan<float>().ToArray(), cam = noiseCamera.AsReadOnlySpan<float>().ToArray(), z0 = z;
        double[] grid = RectifiedFlowSampler.Schedule(steps, rescaleT);
        BudgetLayout? layout = null;
        int budgetTokens = BudgetLayout.SnapTokens(tokens);
        for (int i = 0; i < steps; i++)
        {
            double t = grid[i], tPrev = grid[i + 1];
            bool refresh = relayoutUntil < 0 || i <= relayoutUntil;
            if (layout is null || (refresh && i % relayoutEvery == 0))
            {
                float[] inOctree = new float[z.Length];
                float[] reference = i == 0 ? z : z0;
                for (int k = 0; k < BudgetOctree.Points; k++)
                    Array.Copy(reference, toOctree[k] * lc, inOctree, k * lc, lc);
                (float[] cube, float[] cell) = BudgetOctree.DetailScores(inOctree, lc);
                layout = BudgetLayout.Build(cube, cell, budgetTokens);
            }

            float tModel = (float)(1000.0 * t);
            float[] v = new float[z.Length], vCam = new float[cam.Length];
            using (IDisposable? scope = passScope?.Invoke(tModel))
            {
                using Tensor zT = BudgetHostTensor(z, 1, BudgetOctree.Points, lc), camT = BudgetHostTensor(cam, 1, 1, cc);
                BudgetTokens prepared = model.Prepare(backend, zT, layout);   // once per step: both guidance passes share it
                (Tensor vp, Tensor cp) = model.Forward(backend, prepared, zT, camT, tModel, contextAt is not null ? contextAt(tModel) : positive);
                using (vp)
                using (cp)
                {
                    vp.AsReadOnlySpan<float>().CopyTo(v);
                    cp.AsReadOnlySpan<float>().CopyTo(vCam);
                }

                if (guidance > 0f)
                {
                    (Tensor vn, Tensor cn) = model.Forward(backend, prepared, zT, camT, tModel, contextAt is not null ? contextAt(tModel) : negative!);
                    using (vn)
                    using (cn)
                    {
                        ReadOnlySpan<float> n = vn.AsReadOnlySpan<float>(), nc = cn.AsReadOnlySpan<float>();
                        for (int e = 0; e < v.Length; e++)
                            v[e] = (1f + guidance) * v[e] - guidance * n[e];
                        for (int e = 0; e < vCam.Length; e++)
                            vCam[e] = (1f + guidance) * vCam[e] - guidance * nc[e];
                    }
                }
            }

            float tf = (float)t, dt = (float)(t - tPrev);
            z0 = new float[z.Length];
            for (int e = 0; e < z.Length; e++)
            {
                z0[e] = z[e] - tf * v[e];
                z[e] -= dt * v[e];
            }

            for (int e = 0; e < cam.Length; e++)
                cam[e] -= dt * vCam[e];
            onStep?.Invoke(i + 1);
        }

        Tensor latent = BudgetHostTensor(z, 1, BudgetOctree.Points, lc);
        Tensor camera = BudgetHostTensor(cam, 1, 1, cc);
        return (latent, camera, layout!);
    }

    private static Tensor BudgetHostTensor(float[] data, params long[] shape) => BudgetHost.Make(data, shape);
}
