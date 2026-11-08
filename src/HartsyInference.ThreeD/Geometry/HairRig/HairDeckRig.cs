using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Models.HairDeck;

namespace HartsyInference.ThreeD.Geometry.HairRig;

/// <summary>The rig of one HairDeck sample: the skin of the hair cards and of the chain tubes, and the joints and clips that
/// drive them. Joint 0 is the head; every chain that binds a card adds one joint per segment.</summary>
/// <param name="Rig">Joints and the four motion clips.</param>
/// <param name="HairSkin">Skin of the hair mesh, one entry per vertex.</param>
/// <param name="ChainSkin">Skin of the chain tube mesh, or null when there are no chain tubes.</param>
/// <param name="ChainsBound">Chains that bind at least one card.</param>
/// <param name="CardsOnChains">Cards that follow a chain; the others stay with the head.</param>
public sealed record HairRigResult(HairGlbRig Rig, HairGlbSkin HairSkin, HairGlbSkin? ChainSkin, int ChainsBound, int CardsOnChains);

/// <summary>
/// Builds the rig of a HairDeck sample from its hair cards, chains and tube mesh, following the geometric skinning of
/// HairDeck's <c>hairdeck/spring/sim.py</c> and <c>hairdeck/vrm/prep.py</c>. The pivot of the head motion is the mean of the
/// card roots, and the face width is <see cref="ChainBinding.FaceWidth"/>; the reference takes both from the VRM body, which
/// this export does not have. Chain tubes follow the nearest chain with full chain weight, which the reference does not do.
/// </summary>
public static class HairDeckRig
{
    private const double ChainDrag = 0.4;
    private const int RowStride = 8;
    private const int MaxJoints = ushort.MaxValue + 1;
    private const int TimeCount = SpringBoneSim.Frames + 1;

    /// <summary>Builds the rig; throws when no chain binds a card, since rigged output needs at least one chain.</summary>
    /// <exception cref="InvalidOperationException">No chain has two joints or binds a card.</exception>
    public static HairRigResult Build(IReadOnlyList<HairCard> cards, Mesh hair, IReadOnlyList<DecodedChain> chains, Mesh? chainTubes)
    {
        ArgumentNullException.ThrowIfNull(cards);
        ArgumentNullException.ThrowIfNull(hair);
        ArgumentNullException.ThrowIfNull(chains);

        // Candidate chains: cleaned joints with at least two of them, as in prep().
        List<double[]> candidates = [];
        List<double> stiffness = [];
        foreach (DecodedChain chain in chains)
        {
            double[] cleaned = ChainBinding.Clean(ToDouble(chain.Joints, chain.JointCount * 3));
            if (cleaned.Length >= 6)
            {
                candidates.Add(cleaned);
                stiffness.Add(chain.Stiffness);
            }
        }

        double[] pivot = CardRootMean(cards);
        int vertexBase = 0;
        int[] cardChain = new int[cards.Count];
        List<double[]> cardVertices = new(cards.Count);
        List<(int Segment, double Weight, int Segment2, double Alpha)[]> cardBinding = new(cards.Count);
        for (int ci = 0; ci < cards.Count; ci++)
        {
            double[] sequence = cards[ci].Sequence;
            int n = sequence.Length / RowStride;
            double[] vertices = ToDoubleSlice(hair.Vertices, vertexBase * 3, 4 * n * 3);
            vertexBase += 4 * n;
            cardVertices.Add(vertices);

            double[] control = new double[n * 3];
            for (int i = 0; i < n; i++)
            {
                control[3 * i] = sequence[RowStride * i];
                control[3 * i + 1] = sequence[RowStride * i + 1];
                control[3 * i + 2] = sequence[RowStride * i + 2];
            }

            int chain = ChainBinding.AssignChain(ChainBinding.Centreline(control), candidates, ChainBinding.FaceWidth);
            if (chain >= 0 && MeanLowerDistance(vertices, candidates[chain]) >= ChainBinding.Assign * ChainBinding.FaceWidth)
            {
                chain = -1; // the vertex-based test disagrees with the centreline: the card stays rigid, as in prep()
            }

            cardChain[ci] = chain;
            cardBinding.Add(chain >= 0 ? BindVertices(vertices, candidates[chain]) : []);
        }

        // Chains that bind a card, renumbered in candidate order.
        int[] remap = new int[candidates.Count];
        Array.Fill(remap, -1);
        for (int ci = 0; ci < cards.Count; ci++)
        {
            if (cardChain[ci] >= 0)
            {
                remap[cardChain[ci]] = 0;
            }
        }

        List<double[]> used = [];
        List<double> usedStiffness = [];
        for (int c = 0; c < candidates.Count; c++)
        {
            if (remap[c] == 0)
            {
                remap[c] = used.Count;
                used.Add(candidates[c]);
                usedStiffness.Add(stiffness[c]);
            }
        }

        if (used.Count == 0)
        {
            throw new InvalidOperationException("No spring chain binds a hair card, so there is nothing to rig. Rigged output needs chains.");
        }

        // Joints: 0 is the head; the bones of each used chain follow in chain order.
        int[] boneBase = new int[used.Count];
        int joints = 1;
        for (int u = 0; u < used.Count; u++)
        {
            boneBase[u] = joints;
            joints += used[u].Length / 3 - 1;
        }

        if (joints > MaxJoints)
        {
            throw new InvalidOperationException($"The rig needs {joints} joints; glTF skins are limited to {MaxJoints}.");
        }

        // Skin of the hair vertices (head weight, then the segment weights of the chain the card follows).
        int vertexCount = hair.VertexCount;
        ushort[] hairJoints = new ushort[vertexCount * 4];
        float[] hairWeights = new float[vertexCount * 4];
        int v = 0;
        int cardsOnChains = 0;
        for (int ci = 0; ci < cards.Count; ci++)
        {
            int count = cardVertices[ci].Length / 3;
            int chain = cardChain[ci];
            if (chain >= 0)
            {
                cardsOnChains++;
            }

            for (int i = 0; i < count; i++, v++)
            {
                if (chain < 0)
                {
                    SetInfluences(hairJoints, hairWeights, v, [(0, 1.0)]);
                    continue;
                }

                int u = remap[chain];
                (int segment, double weight, int segment2, double alpha) = cardBinding[ci][i];
                SetInfluences(hairJoints, hairWeights, v, BlendInfluences(0, boneBase[u], weight, segment, segment2, alpha));
            }
        }

        // Skin of the chain tubes: each vertex follows the nearest used chain with full chain weight.
        HairGlbSkin? chainSkin = null;
        if (chainTubes is not null && chainTubes.VertexCount > 0)
        {
            int tubeCount = chainTubes.VertexCount;
            ushort[] tubeJoints = new ushort[tubeCount * 4];
            float[] tubeWeights = new float[tubeCount * 4];
            double[] tubePoints = ToDouble(chainTubes.Vertices, tubeCount * 3);
            for (int i = 0; i < tubeCount; i++)
            {
                double[] point = [tubePoints[3 * i], tubePoints[3 * i + 1], tubePoints[3 * i + 2]];
                int best = -1;
                double bestDistance = double.PositiveInfinity;
                for (int u = 0; u < used.Count; u++)
                {
                    ChainBinding.Projection p = ChainBinding.Project(point, used[u])[0];
                    if (p.Distance < bestDistance)
                    {
                        bestDistance = p.Distance;
                        best = u;
                    }
                }

                (int segment, double unusedWeight, int segment2, double alpha) = BindVertices(point, used[best])[0];
                SetInfluences(tubeJoints, tubeWeights, i, BlendInfluences(-1, boneBase[best], 1.0, segment, segment2, alpha));
            }

            chainSkin = new HairGlbSkin(tubeJoints, tubeWeights);
        }

        HairGlbRig rig = BuildRig(pivot, used, usedStiffness, boneBase, joints);
        return new HairRigResult(rig, new HairGlbSkin(hairJoints, hairWeights), chainSkin, used.Count, cardsOnChains);
    }

    /// <summary>The joints (rest pose) and the four motion clips of the rig.</summary>
    private static HairGlbRig BuildRig(double[] pivot, IReadOnlyList<double[]> used, IReadOnlyList<double> usedStiffness,
        int[] boneBase, int jointCount)
    {
        List<HairGlbJoint> joints = new(jointCount);
        joints.Add(new HairGlbJoint("hair_head", ToFloat(pivot), [0f, 0f, 0f, 1f], TranslationMatrix(-pivot[0], -pivot[1], -pivot[2])));
        for (int u = 0; u < used.Count; u++)
        {
            double[] chain = used[u];
            for (int k = 0; k < chain.Length / 3 - 1; k++)
            {
                double[] at = [chain[3 * k], chain[3 * k + 1], chain[3 * k + 2]];
                joints.Add(new HairGlbJoint($"hair_chain{u}_bone{k}", ToFloat(at), [0f, 0f, 0f, 1f],
                    TranslationMatrix(-at[0], -at[1], -at[2])));
            }
        }

        List<SpringBoneSim.Chain> simChains = [];
        for (int u = 0; u < used.Count; u++)
        {
            simChains.Add(new SpringBoneSim.Chain(used[u], usedStiffness[u], ChainDrag, new double[3]));
        }

        float[] times = new float[TimeCount];
        for (int f = 1; f < TimeCount; f++)
        {
            times[f] = (float)(f / (double)SpringBoneSim.Fps);
        }

        List<HairGlbAnimation> animations = [];
        foreach (string name in SpringBoneSim.MotionNames)
        {
            SpringBoneSim.Motion motion = SpringBoneSim.GetMotion(name);
            SpringBoneSim.ChainState[] states = SpringBoneSim.Simulate(simChains, motion, pivot);
            List<HairGlbChannel> channels = new(jointCount);

            // Head: the rigid motion about the pivot, R (x - pivot) + pivot + T.
            float[] headT = new float[TimeCount * 3];
            float[] headR = new float[TimeCount * 4];
            WriteTranslation(headT, 0, pivot);
            WriteQuaternion(headR, 0, Identity());
            for (int f = 0; f < SpringBoneSim.Frames; f++)
            {
                double[] p = [pivot[0] + motion.Translations[3 * f], pivot[1] + motion.Translations[3 * f + 1], pivot[2] + motion.Translations[3 * f + 2]];
                WriteTranslation(headT, (f + 1) * 3, p);
                WriteQuaternion(headR, (f + 1) * 4, Matrix(motion.Rotations, f * 9));
            }

            channels.Add(new HairGlbChannel(headT, headR));

            // Bones: the simulated head position and rest-to-current rotation of each segment.
            for (int u = 0; u < used.Count; u++)
            {
                SpringBoneSim.ChainState state = states[u];
                double[] chain = used[u];
                for (int k = 0; k < state.Bones; k++)
                {
                    float[] t = new float[TimeCount * 3];
                    float[] r = new float[TimeCount * 4];
                    WriteTranslation(t, 0, [chain[3 * k], chain[3 * k + 1], chain[3 * k + 2]]);
                    WriteQuaternion(r, 0, Identity());
                    for (int f = 0; f < SpringBoneSim.Frames; f++)
                    {
                        int at = (f * state.Bones + k);
                        WriteTranslation(t, (f + 1) * 3, [state.Heads[3 * at], state.Heads[3 * at + 1], state.Heads[3 * at + 2]]);
                        WriteQuaternion(r, (f + 1) * 4, Matrix(state.Rotations, at * 9));
                    }

                    channels.Add(new HairGlbChannel(t, r));
                }
            }

            animations.Add(new HairGlbAnimation(name, times, channels));
        }

        return new HairGlbRig(joints, animations);
    }

    /// <summary>Per vertex: the segment, weight, neighbouring segment and blend factor on one chain.</summary>
    private static (int Segment, double Weight, int Segment2, double Alpha)[] BindVertices(double[] vertices, double[] chain)
    {
        (int[] segment, double[] weight, int[] segment2, double[] alpha) = ChainBinding.Bind(vertices, chain, ChainBinding.FaceWidth);
        (int, double, int, double)[] result = new (int, double, int, double)[segment.Length];
        for (int i = 0; i < segment.Length; i++)
        {
            result[i] = (segment[i], weight[i], segment2[i], alpha[i]);
        }

        return result;
    }

    /// <summary>The influences of one vertex: the head (joint <paramref name="headJoint"/>) with weight 1 - w, and the segment
    /// joints with w * alpha and w * (1 - alpha); a head index of -1 gives no head influence.</summary>
    private static List<(int Joint, double Weight)> BlendInfluences(int headJoint, int boneBase, double w, int segment, int segment2,
        double alpha)
    {
        List<(int Joint, double Weight)> influences = [];
        if (headJoint >= 0 && w < 1.0)
        {
            influences.Add((headJoint, 1.0 - w));
        }

        int first = boneBase + segment;
        int second = boneBase + segment2;
        if (second == first)
        {
            influences.Add((first, w));
        }
        else
        {
            influences.Add((first, w * alpha));
            influences.Add((second, w * (1.0 - alpha)));
        }

        return influences;
    }

    /// <summary>Writes up to four influences of a vertex; zero weights are dropped, the rest are normalised to sum to 1.</summary>
    private static void SetInfluences(ushort[] joints, float[] weights, int vertex, IEnumerable<(int Joint, double Weight)> influences)
    {
        List<(int Joint, double Weight)> kept = [];
        foreach ((int joint, double weight) in influences)
        {
            if (weight > 1e-9)
            {
                kept.Add((joint, weight));
            }
        }

        double sum = 0;
        foreach ((int _, double weight) in kept)
        {
            sum += weight;
        }

        if (kept.Count == 0 || sum <= 0)
        {
            kept.Add((0, 1.0));
            sum = 1.0;
        }

        for (int i = 0; i < 4; i++)
        {
            joints[4 * vertex + i] = i < kept.Count ? (ushort)kept[i].Joint : (ushort)0;
            weights[4 * vertex + i] = i < kept.Count ? (float)(kept[i].Weight / sum) : 0f;
        }
    }

    private static double[] CardRootMean(IReadOnlyList<HairCard> cards)
    {
        double[] sum = new double[3];
        foreach (HairCard card in cards)
        {
            for (int c = 0; c < 3; c++)
            {
                sum[c] += card.Sequence[c];
            }
        }

        for (int c = 0; c < 3; c++)
        {
            sum[c] /= cards.Count;
        }

        return sum;
    }

    private static double MeanLowerDistance(double[] vertices, double[] chain)
    {
        int m = vertices.Length / 3;
        double[] lower = vertices[(3 * (int)(0.4 * m))..];
        ChainBinding.Projection[] projections = ChainBinding.Project(lower, chain);
        double sum = 0;
        foreach (ChainBinding.Projection p in projections)
        {
            sum += p.Distance;
        }

        return sum / projections.Length;
    }

    private static double[] ToDouble(float[] values, int count)
    {
        double[] result = new double[count];
        for (int i = 0; i < count; i++)
        {
            result[i] = values[i];
        }

        return result;
    }

    private static double[] ToDoubleSlice(float[] values, int start, int count) => ToDouble(values[start..(start + count)], count);

    private static float[] ToFloat(double[] values) => [(float)values[0], (float)values[1], (float)values[2]];

    private static double[] Identity() => [1, 0, 0, 0, 1, 0, 0, 0, 1];

    private static double[] Matrix(double[] values, int offset) => values[offset..(offset + 9)];

    private static void WriteTranslation(float[] dst, int offset, double[] p)
    {
        dst[offset] = (float)p[0];
        dst[offset + 1] = (float)p[1];
        dst[offset + 2] = (float)p[2];
    }

    /// <summary>Writes the quaternion (x, y, z, w) of a row-major rotation matrix, normalised.</summary>
    private static void WriteQuaternion(float[] dst, int offset, double[] m)
    {
        double trace = m[0] + m[4] + m[8];
        double w, x, y, z;
        if (trace > 0)
        {
            double s = Math.Sqrt(trace + 1) * 2;
            w = 0.25 * s;
            x = (m[7] - m[5]) / s;
            y = (m[2] - m[6]) / s;
            z = (m[3] - m[1]) / s;
        }
        else if (m[0] > m[4] && m[0] > m[8])
        {
            double s = Math.Sqrt(1 + m[0] - m[4] - m[8]) * 2;
            w = (m[7] - m[5]) / s;
            x = 0.25 * s;
            y = (m[1] + m[3]) / s;
            z = (m[2] + m[6]) / s;
        }
        else if (m[4] > m[8])
        {
            double s = Math.Sqrt(1 + m[4] - m[0] - m[8]) * 2;
            w = (m[2] - m[6]) / s;
            x = (m[1] + m[3]) / s;
            y = 0.25 * s;
            z = (m[5] + m[7]) / s;
        }
        else
        {
            double s = Math.Sqrt(1 + m[8] - m[0] - m[4]) * 2;
            w = (m[3] - m[1]) / s;
            x = (m[2] + m[6]) / s;
            y = (m[5] + m[7]) / s;
            z = 0.25 * s;
        }

        double norm = Math.Sqrt(x * x + y * y + z * z + w * w);
        dst[offset] = (float)(x / norm);
        dst[offset + 1] = (float)(y / norm);
        dst[offset + 2] = (float)(z / norm);
        dst[offset + 3] = (float)(w / norm);
    }

    /// <summary>Column-major 4x4 translation matrix.</summary>
    private static float[] TranslationMatrix(double x, double y, double z) =>
    [
        1f, 0f, 0f, 0f,
        0f, 1f, 0f, 0f,
        0f, 0f, 1f, 0f,
        (float)x, (float)y, (float)z, 1f,
    ];
}
