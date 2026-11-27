using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.HairDeck;

/// <summary>Decodes the 27-channel chain tokens (<c>hairdeck/spring/chaindit.py</c> <c>decode</c>): eight arc-length samples
/// of the path, the joint count and the stiffness, standardised by the chain statistics; the joints are the path resampled
/// at <c>n</c> arc-length positions. The reference's numpy steps are computed in double precision.</summary>
public static class HairDeckChainDecoder
{
    /// <summary>Chain channels: eight path points, joint count, stiffness, and the existence flag.</summary>
    public const int TokenDim = 27;

    /// <summary>Arc-length samples of the chain path.</summary>
    public const int PathSamples = 8;

    private const int ValueCount = TokenDim - 1;
    private const int CountAt = PathSamples * 3;
    private const int StiffnessAt = CountAt + 1;
    private const int MinJoints = 2;
    private const int MaxJoints = 20;

    /// <summary>Decodes the existing chain slots of <paramref name="tokens"/> <c>[slots, 27]</c>.</summary>
    public static IReadOnlyList<DecodedChain> Decode(Tensor tokens, ReadOnlySpan<float> mu, ReadOnlySpan<float> sd)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        if (tokens.Shape.Rank != 2 || tokens.Shape[1] != TokenDim || tokens.DType != DType.F32)
            throw new ArgumentException($"HairDeck chain tokens must be F32 [slots, {TokenDim}], got {tokens.Shape}.", nameof(tokens));
        if (mu.Length != ValueCount || sd.Length != ValueCount)
            throw new ArgumentException($"HairDeck chain statistics have {ValueCount} channels.");

        List<DecodedChain> chains = [];
        unsafe
        {
            float* p = (float*)tokens.DataPointer;
            for (int slot = 0; slot < tokens.Shape[0]; slot++)
            {
                float* row = p + (long)slot * TokenDim;
                if (row[TokenDim - 1] <= 0f) continue;
                chains.Add(DecodeChain(row, mu, sd));
            }
        }
        return chains;
    }

    private static unsafe DecodedChain DecodeChain(float* row, ReadOnlySpan<float> mu, ReadOnlySpan<float> sd)
    {
        Span<double> v = stackalloc double[ValueCount];
        for (int k = 0; k < ValueCount; k++) v[k] = row[k] * (double)sd[k] + mu[k];

        double[] path = new double[PathSamples * 3];
        for (int k = 0; k < path.Length; k++) path[k] = v[k];

        double[] arc = new double[PathSamples];
        for (int i = 1; i < PathSamples; i++)
        {
            double dx = path[i * 3] - path[(i - 1) * 3];
            double dy = path[i * 3 + 1] - path[(i - 1) * 3 + 1];
            double dz = path[i * 3 + 2] - path[(i - 1) * 3 + 2];
            arc[i] = arc[i - 1] + Math.Sqrt(dx * dx + dy * dy + dz * dz);
        }

        int joints = (int)Math.Clamp(Math.Round(v[CountAt], MidpointRounding.ToEven), MinJoints, MaxJoints);
        double total = arc[PathSamples - 1];
        double[] result = new double[joints * 3];
        for (int j = 0; j < joints; j++)
        {
            double at = j == joints - 1 ? total : total * j / (joints - 1);
            for (int c = 0; c < 3; c++) result[j * 3 + c] = Interpolate(arc, path, c, at);
        }

        float[] jointValues = new float[result.Length];
        for (int k = 0; k < result.Length; k++) jointValues[k] = (float)result[k];
        float[] pathValues = new float[path.Length];
        for (int k = 0; k < path.Length; k++) pathValues[k] = (float)path[k];
        float stiffness = (float)Math.Clamp(v[StiffnessAt], 0.0, 1.0);
        return new DecodedChain(jointValues, joints, stiffness) { Path = pathValues };
    }

    /// <summary>numpy.interp of channel <paramref name="channel"/> at <paramref name="x"/> over the increasing abscissae <paramref name="xp"/>.</summary>
    private static double Interpolate(double[] xp, double[] path, int channel, double x)
    {
        int last = xp.Length - 1;
        if (x <= xp[0]) return path[channel];
        if (x >= xp[last]) return path[last * 3 + channel];
        int i = 0;
        while (i < last - 1 && xp[i + 1] <= x) i++;
        double slope = (path[(i + 1) * 3 + channel] - path[i * 3 + channel]) / (xp[i + 1] - xp[i]);
        return slope * (x - xp[i]) + path[i * 3 + channel];
    }
}
