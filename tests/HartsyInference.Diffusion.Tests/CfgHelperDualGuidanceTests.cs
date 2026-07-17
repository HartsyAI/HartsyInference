using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Unit coverage for <see cref="CfgHelper.ApplyDualCfg"/> — the shared OmniGen2 / Boogu-Image dual
/// (text + image) guidance combine. Verifies both published forms of the formula agree: the OmniGen2 form
/// <c>uncond + ig·(mid − uncond) + tg·(cond − mid)</c> and Boogu's algebraically-equal
/// <c>cond + (tg−1)·(cond − mid) + (ig−1)·(mid − uncond)</c>.</summary>
public sealed class CfgHelperDualGuidanceTests
{
    private static unsafe Tensor MakeTensor(float[] values)
    {
        Tensor t = new(new TensorShape(1, values.Length), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < values.Length; i++) p[i] = values[i];
        return t;
    }

    [Fact]
    public unsafe void ApplyDualCfg_MatchesOmniGen2AndBooguForms()
    {
        float[] condV = [0.5f, -1.25f, 3.0f, 0.0f, 2.5f];
        float[] midV = [0.25f, -0.75f, 2.0f, 1.0f, -0.5f];
        float[] uncondV = [-0.5f, 0.5f, 1.0f, -2.0f, 0.75f];
        const float tg = 5.0f;
        const float ig = 2.0f;

        using Tensor cond = MakeTensor(condV);
        using Tensor mid = MakeTensor(midV);
        using Tensor uncond = MakeTensor(uncondV);

        using Tensor result = CfgHelper.ApplyDualCfg(cond, mid, uncond, tg, ig);
        float* r = (float*)result.DataPointer;

        for (int i = 0; i < condV.Length; i++)
        {
            float omniGen2Form = uncondV[i] + ig * (midV[i] - uncondV[i]) + tg * (condV[i] - midV[i]);
            float booguForm = condV[i] + (tg - 1f) * (condV[i] - midV[i]) + (ig - 1f) * (midV[i] - uncondV[i]);
            Assert.Equal(omniGen2Form, booguForm, 4);
            Assert.Equal(omniGen2Form, r[i], 4);
        }
    }

    [Fact]
    public unsafe void ApplyDualCfg_UnitScales_ReduceToCond()
    {
        float[] condV = [1.5f, -0.5f, 0.25f];
        using Tensor cond = MakeTensor(condV);
        using Tensor mid = MakeTensor([0.1f, 0.2f, 0.3f]);
        using Tensor uncond = MakeTensor([-1.0f, 2.0f, 0.0f]);

        // tg = ig = 1: uncond + (mid − uncond) + (cond − mid) = cond.
        using Tensor result = CfgHelper.ApplyDualCfg(cond, mid, uncond, 1.0f, 1.0f);
        float* r = (float*)result.DataPointer;
        for (int i = 0; i < condV.Length; i++)
            Assert.Equal(condV[i], r[i], 5);
    }

    [Fact]
    public void ApplyDualCfg_ShapeMismatch_Throws()
    {
        using Tensor cond = MakeTensor([1f, 2f, 3f]);
        using Tensor mid = MakeTensor([1f, 2f, 3f]);
        using Tensor uncond = MakeTensor([1f, 2f]);
        Assert.Throws<ArgumentException>(() => CfgHelper.ApplyDualCfg(cond, mid, uncond, 4f, 2f));
    }
}
