using System.Buffers.Binary;
using System.Text.Json;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.HairRig;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Models.HairDeck;

namespace HartsyInference.ThreeD.Tests;

/// <summary>Spring-bone simulation, chain binding and the rigged GLB of the hair rig.</summary>
public sealed class HairRigTests
{
    private static readonly float[] HairSrgb = [0.4f, 0.3f, 0.25f];

    [Fact]
    public void Simulation_matches_the_reference_python_for_turn_and_jump()
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "hairdeck_sim_reference.json")));
        JsonElement root = doc.RootElement;
        double[] pivot = Flatten(root.GetProperty("piv"));
        List<SpringBoneSim.Chain> chains = [];
        foreach (JsonElement chain in root.GetProperty("chains").EnumerateArray())
        {
            chains.Add(new SpringBoneSim.Chain(Flatten(chain.GetProperty("J")), chain.GetProperty("stiff").GetDouble(),
                chain.GetProperty("drag").GetDouble(), Flatten(chain.GetProperty("grav"))));
        }

        foreach (string name in new[] { "turn", "jump" })
        {
            JsonElement motion = root.GetProperty("motions").GetProperty(name);
            SpringBoneSim.Motion sim = SpringBoneSim.GetMotion(name);
            SpringBoneSim.ChainState[] states = SpringBoneSim.Simulate(chains, sim, pivot);
            foreach (string key in new[] { "0", "10", "75", "149" })
            {
                int f = int.Parse(key);
                JsonElement frame = motion.GetProperty("frames").GetProperty(key);
                for (int c = 0; c < chains.Count; c++)
                {
                    JsonElement expected = frame[c];
                    double[] heads = Flatten(expected.GetProperty("heads"));
                    double[] rots = Flatten(expected.GetProperty("rots"));
                    SpringBoneSim.ChainState state = states[c];
                    for (int i = 0; i < heads.Length; i++)
                    {
                        Assert.InRange(state.Heads[f * heads.Length + i] - heads[i], -1e-9, 1e-9);
                    }
                    for (int i = 0; i < rots.Length; i++)
                    {
                        Assert.InRange(state.Rotations[f * rots.Length + i] - rots[i], -1e-9, 1e-9);
                    }
                }
            }
        }
    }

    [Fact]
    public void Clean_drops_a_repeated_joint_and_keeps_the_first()
    {
        double[] cleaned = ChainBinding.Clean([0, 0, 0, 0, 0, 0, 1, 0, 0]);
        Assert.Equal(new double[] { 0, 0, 0, 1, 0, 0 }, cleaned);
    }

    [Fact]
    public void Blend_follows_the_neighbour_across_a_joint()
    {
        double[] chain = [0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0];
        // A point at u = 0.3 of segment 1 (x from 1 to 2) blends toward segment 0 with alpha = 0.5 + u.
        (int[] Segment, double[] Weight, int[] Segment2, double[] Alpha) bound = ChainBinding.Bind([1.3, 0, 0], chain, ChainBinding.FaceWidth);
        Assert.Equal(1, bound.Segment[0]);
        Assert.Equal(0, bound.Segment2[0]);
        Assert.Equal(0.8, bound.Alpha[0], 9);
    }

    [Fact]
    public void Weight_ramps_from_the_root()
    {
        double[] chain = [0, 0, 0, 1, 0, 0];
        double ramp = ChainBinding.Ramp * ChainBinding.FaceWidth;
        (int[] Segment, double[] Weight, int[] Segment2, double[] Alpha) root = ChainBinding.Bind([0, 0, 0], chain, ChainBinding.FaceWidth);
        (int[] Segment, double[] Weight, int[] Segment2, double[] Alpha) past = ChainBinding.Bind([10 * ramp, 0, 0], chain, ChainBinding.FaceWidth);
        Assert.Equal(0.0, root.Weight[0], 9);
        Assert.Equal(1.0, past.Weight[0], 9);
    }

    [Fact]
    public void Rest_pose_skins_back_to_the_rest_positions()
    {
        (HairRigResult result, Mesh hair) = BuildSample();
        HairGlbRig rig = result.Rig;
        for (int v = 0; v < hair.VertexCount; v++)
        {
            double[] skinned = Skin(rig, result.HairSkin, v, clip: 0, frame: 0, hair);
            Assert.Equal(hair.Vertices[3 * v], skinned[0], 5);
            Assert.Equal(hair.Vertices[3 * v + 1], skinned[1], 5);
            Assert.Equal(hair.Vertices[3 * v + 2], skinned[2], 5);
        }
    }

    [Fact]
    public void Turn_moves_the_hair_as_the_simulation_says()
    {
        (HairRigResult result, Mesh hair) = BuildSample();
        Assert.Equal(3, result.CardsOnChains);
        // The pivot is the mean of the card roots: x = 0.00, 0.01, 0.02 (mean 0.01) at y = 0.40, z = 0.
        double[] pivot = [0.01, 0.40, 0.0];
        SpringBoneSim.Motion turn = SpringBoneSim.GetMotion("turn");
        double[] joints = ChainJoints();
        SpringBoneSim.ChainState state = SpringBoneSim.Simulate([new SpringBoneSim.Chain(joints, 0.5, 0.4, new double[3])], turn, pivot)[0];
        int frame = SpringBoneSim.Frames - 1;
        (int[] Segment, double[] Weight, int[] Segment2, double[] Alpha) binding = ChainBinding.Bind(Positions(hair), joints, ChainBinding.FaceWidth);

        for (int v = 0; v < hair.VertexCount; v++)
        {
            double[] rest = [hair.Vertices[3 * v], hair.Vertices[3 * v + 1], hair.Vertices[3 * v + 2]];
            double[] rigid = Rigid(rest, turn, pivot, frame);
            double[] follow = Follow(rest, joints, state, frame, binding.Segment[v], binding.Segment2[v], binding.Alpha[v]);
            double w = binding.Weight[v];
            double[] expected = new int[] { 0, 1, 2 }.Select(c => (1 - w) * rigid[c] + w * follow[c]).ToArray();

            double[] skinned = Skin(result.Rig, result.HairSkin, v, clip: 0, frame: frame + 1, hair);
            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(expected[i], skinned[i], 4);
            }
        }
    }

    [Fact]
    public void Clips_hold_one_channel_per_joint_and_four_motions()
    {
        (HairRigResult result, _) = BuildSample();
        Assert.Equal(SpringBoneSim.MotionNames, result.Rig.Animations.Select(a => a.Name).ToArray());
        foreach (HairGlbAnimation clip in result.Rig.Animations)
        {
            Assert.Equal(result.Rig.Joints.Count, clip.Channels.Count);
            Assert.Equal(SpringBoneSim.Frames + 1, clip.Times.Length);
        }
    }

    [Fact]
    public void Rigged_glb_has_a_skin_and_the_four_animations()
    {
        (HairRigResult result, Mesh hair) = BuildSample();
        byte[] glb = HairCardGlbWriter.Write(
            [new HairGlbNode("hair_cards", hair, new HairGlbMaterial("hair", [1f, 1f, 1f, 1f], 0f, 0.75f, DoubleSided: true), result.HairSkin)],
            result.Rig);
        JsonElement json = ReadJson(glb);
        Assert.Single(json.GetProperty("skins").EnumerateArray());
        Assert.Equal(result.Rig.Joints.Count, json.GetProperty("skins")[0].GetProperty("joints").GetArrayLength());
        Assert.Equal(4, json.GetProperty("animations").GetArrayLength());
        Assert.Equal(0, json.GetProperty("nodes")[0].GetProperty("skin").GetInt32());
    }

    [Fact]
    public void Build_fails_when_no_chain_binds_a_card()
    {
        (IReadOnlyList<HairCard> cards, Mesh hair) = Cards(xOffset: 5.0);
        DecodedChain far = new(FloatChain(), 4, 0.5f);
        Assert.Throws<InvalidOperationException>(() => HairDeckRig.Build(cards, hair, [far], null));
    }

    [Fact]
    public void Card_whose_outer_vertices_are_far_from_the_chain_stays_rigid()
    {
        // The two narrow cards hang beside the chain. The wide card's centreline is on the chain, but the mean distance of
        // its lower vertices (offset 0.2 m from the centreline) is over the assignment limit, so it stays with the head.
        List<HairCard> cards = [];
        foreach ((double x, double width) in new[] { (0.0, 0.02), (0.01, 0.02), (0.0, 0.4) })
        {
            double[] sequence = new double[4 * 8];
            for (int i = 0; i < 4; i++)
            {
                sequence[8 * i] = x;
                sequence[8 * i + 1] = 0.40 - 0.05 * i;
                sequence[8 * i + 2] = 0.0;
                sequence[8 * i + 3] = width;
                sequence[8 * i + 4] = 0.005;
                sequence[8 * i + 7] = 1.0;
            }
            cards.Add(new HairCard(sequence, 0, null));
        }

        Mesh hair = HairCardMesher.Build(cards, HairSrgb)!.Mesh;
        HairRigResult result = HairDeckRig.Build(cards, hair, [new DecodedChain(FloatChain(), 4, 0.5f)], null);

        Assert.Equal(2, result.CardsOnChains);
        for (int v = 2 * 16; v < hair.VertexCount; v++)
        {
            Assert.Equal(0, result.HairSkin.JointIndices[4 * v]);
            Assert.Equal(1f, result.HairSkin.Weights[4 * v]);
        }
    }

    private static (HairRigResult Result, Mesh Hair) BuildSample()
    {
        (IReadOnlyList<HairCard> cards, Mesh hair) = Cards(xOffset: 0.0);
        DecodedChain chain = new(FloatChain(), 4, 0.5f);
        return (HairDeckRig.Build(cards, hair, [chain], null), hair);
    }

    /// <summary>Three cards hanging from the scalp beside a vertical chain at x = 0.</summary>
    private static (IReadOnlyList<HairCard> Cards, Mesh Hair) Cards(double xOffset)
    {
        List<HairCard> cards = [];
        for (int c = 0; c < 3; c++)
        {
            double[] sequence = new double[4 * 8];
            for (int i = 0; i < 4; i++)
            {
                sequence[8 * i] = xOffset + 0.01 * c;
                sequence[8 * i + 1] = 0.40 - 0.05 * i;
                sequence[8 * i + 2] = 0.0;
                sequence[8 * i + 3] = 0.02;
                sequence[8 * i + 4] = 0.005;
                sequence[8 * i + 5] = 0.0;
                sequence[8 * i + 6] = 0.0;
                sequence[8 * i + 7] = 1.0;
            }
            cards.Add(new HairCard(sequence, 0, null));
        }

        HairCardMesh mesh = HairCardMesher.Build(cards, HairSrgb)!;
        return (cards, mesh.Mesh);
    }

    private static float[] FloatChain() => [0f, 0.42f, 0f, 0f, 0.36f, 0f, 0f, 0.30f, 0f, 0f, 0.24f, 0f];

    private static double[] ChainJoints() => [0, 0.42, 0, 0, 0.36, 0, 0, 0.30, 0, 0, 0.24, 0];

    private static double[] Positions(Mesh hair)
    {
        double[] p = new double[hair.VertexCount * 3];
        for (int i = 0; i < p.Length; i++)
        {
            p[i] = hair.Vertices[i];
        }

        return p;
    }

    /// <summary>The chain part of a point's position: a blend of the two neighbouring bones' simulated follows, each
    /// head + rotation * (rest - joint), as the reference's deform() gives it.</summary>
    private static double[] Follow(double[] rest, double[] joints, SpringBoneSim.ChainState state, int frame, int k, int k2, double alpha)
    {
        double[] a = BoneFollow(rest, joints, state, frame, k);
        double[] b = BoneFollow(rest, joints, state, frame, k2);
        return new int[] { 0, 1, 2 }.Select(c => alpha * a[c] + (1 - alpha) * b[c]).ToArray();
    }

    private static double[] BoneFollow(double[] rest, double[] joints, SpringBoneSim.ChainState state, int frame, int bone)
    {
        int at = frame * state.Bones + bone;
        double[] rot = Matrix(state.Rotations, at * 9);
        double[] head = [state.Heads[3 * at], state.Heads[3 * at + 1], state.Heads[3 * at + 2]];
        double[] local = [rest[0] - joints[3 * bone], rest[1] - joints[3 * bone + 1], rest[2] - joints[3 * bone + 2]];
        double[] moved = Apply(rot, local);
        return [moved[0] + head[0], moved[1] + head[1], moved[2] + head[2]];
    }

    private static double[] Rigid(double[] rest, SpringBoneSim.Motion motion, double[] pivot, int frame)
    {
        double[] r = Matrix(motion.Rotations, frame * 9);
        double[] local = [rest[0] - pivot[0], rest[1] - pivot[1], rest[2] - pivot[2]];
        double[] rotated = Apply(r, local);
        return [rotated[0] + pivot[0] + motion.Translations[3 * frame], rotated[1] + pivot[1] + motion.Translations[3 * frame + 1],
            rotated[2] + pivot[2] + motion.Translations[3 * frame + 2]];
    }

    /// <summary>glTF linear blend skinning of vertex <paramref name="v"/>: sum over its joints of weight * (world * inverse bind) * position.</summary>
    private static double[] Skin(HairGlbRig rig, HairGlbSkin skin, int v, int clip, int frame, Mesh hair)
    {
        double[] position = [hair.Vertices[3 * v], hair.Vertices[3 * v + 1], hair.Vertices[3 * v + 2]];
        double[] sum = new double[3];
        for (int i = 0; i < 4; i++)
        {
            double weight = skin.Weights[4 * v + i];
            if (weight == 0)
            {
                continue;
            }

            int j = skin.JointIndices[4 * v + i];
            double[] world = WorldMatrix(rig.Animations[clip].Channels[j], frame);
            double[] inverse = Column4(rig.Joints[j].InverseBindMatrix);
            double[] joint = Multiply4(world, inverse);
            double[] p = ApplyAffine(joint, position);
            for (int c = 0; c < 3; c++)
            {
                sum[c] += weight * p[c];
            }
        }

        return sum;
    }

    private static double[] WorldMatrix(HairGlbChannel channel, int frame)
    {
        double[] m = Identity4();
        double[] q = [channel.Rotations[4 * frame], channel.Rotations[4 * frame + 1], channel.Rotations[4 * frame + 2], channel.Rotations[4 * frame + 3]];
        double x = q[0], y = q[1], z = q[2], w = q[3];
        m[0] = 1 - 2 * (y * y + z * z);
        m[1] = 2 * (x * y - z * w);
        m[2] = 2 * (x * z + y * w);
        m[4] = 2 * (x * y + z * w);
        m[5] = 1 - 2 * (x * x + z * z);
        m[6] = 2 * (y * z - x * w);
        m[8] = 2 * (x * z - y * w);
        m[9] = 2 * (y * z + x * w);
        m[10] = 1 - 2 * (x * x + y * y);
        m[3] = channel.Translations[3 * frame];
        m[7] = channel.Translations[3 * frame + 1];
        m[11] = channel.Translations[3 * frame + 2];
        return m;
    }

    /// <summary>Row-major 4x4 from a glTF column-major matrix (element (row r, column c) is at 4c + r).</summary>
    private static double[] Column4(float[] column)
    {
        double[] m = new double[16];
        for (int r = 0; r < 4; r++)
        {
            for (int c = 0; c < 4; c++)
            {
                m[4 * r + c] = column[4 * c + r];
            }
        }

        return m;
    }

    private static double[] Identity4() => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

    /// <summary>Row-major 4x4 product (inputs in row-major order, glTF's column-major matrices are transposed on load).</summary>
    private static double[] Multiply4(double[] a, double[] b)
    {
        double[] r = new double[16];
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                for (int k = 0; k < 4; k++)
                {
                    r[4 * i + j] += a[4 * i + k] * b[4 * k + j];
                }
            }
        }

        return r;
    }

    private static double[] ApplyAffine(double[] m, double[] p) =>
        [m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3], m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7], m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11]];

    private static double[] Apply(double[] m, double[] v) =>
        [m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2], m[6] * v[0] + m[7] * v[1] + m[8] * v[2]];

    private static double[] Matrix(double[] values, int offset) => values[offset..(offset + 9)];

    private static double[] Flatten(JsonElement element)
    {
        List<double> values = [];
        Walk(element, values);
        return values.ToArray();
    }

    private static void Walk(JsonElement element, List<double> values)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Array:
                foreach (JsonElement child in element.EnumerateArray())
                    Walk(child, values);
                break;
            case JsonValueKind.Number:
                values.Add(element.GetDouble());
                break;
        }
    }

    private static JsonElement ReadJson(byte[] glb)
    {
        int length = (int)BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(12));
        return JsonDocument.Parse(glb.AsMemory(20, length)).RootElement.Clone();
    }
}
