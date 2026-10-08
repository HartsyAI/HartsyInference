using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Moe;

/// <summary>Topology contracts: validation, capability discovery, router lowering, clamp semantics and fingerprints.</summary>
public sealed class SparseTopologyTests
{
    private const int Hidden = 64;

    private static ExpertDescriptor Expert(int inter = 32, DType? dtype = null) => new(Hidden, inter, dtype ?? DType.F32);

    private static MoeLayerDescriptor Layer(int experts = 8, int topK = 2, MoeRouteScoring scoring = MoeRouteScoring.Softmax,
        ExpertGroupDescriptor? shared = null, ExpertProgram? program = null, int groups = 0, int groupsKept = 0,
        int topKPrefill = -1, bool tokenKind = false, ExpertGroupDescriptor? routed = null) =>
        new MoeLayerDescriptor(
            new RouterDescriptor(experts, topK, topKPrefill < 0 ? topK : topKPrefill, scoring, GroupCount: groups, GroupsKept: groupsKept,
                Renormalize: true, HasTokenKindBias: tokenKind).Validated(),
            routed ?? new ExpertGroupDescriptor(experts, Expert()).Validated(),
            shared,
            SharedIsGated: false,
            program ?? ExpertProgram.Swiglu).Validated();

    private static SparseModelTopology Uniform(int layers, Func<int, MoeLayerDescriptor?> moe, Func<int, SequenceStateKind>? state = null) =>
        new(Hidden, Enumerable.Range(0, layers).Select(i => new SparseLayerDescriptor(i, moe(i), state?.Invoke(i) ?? SequenceStateKind.StandardKv)).ToList());

    [Fact]
    public void MixtralStyle_AllLayersSparseSoftmaxUniform()
    {
        SparseModelTopology topology = Uniform(4, _ => Layer(experts: 8, topK: 2));

        SparseCapabilities caps = topology.Capabilities;
        Assert.True(caps.HasSparseExperts);
        Assert.False(caps.HasDenseLayers);
        Assert.False(caps.HasSharedExperts);
        Assert.False(caps.VariableExpertsPerLayer);
        Assert.False(caps.VariableTopK);
        Assert.False(caps.HeterogeneousExpertShapes);
        Assert.False(caps.HasDraftLayers);
        Assert.False(caps.RequiresHostRouting);
        Assert.Equal(new[] { MoeRouteScoring.Softmax }, caps.ScoringKinds);
        Assert.Equal(new[] { SequenceStateKind.StandardKv }, caps.StateKinds);
        Assert.Equal(4, topology.SparseLayerCount);
        Assert.Equal(8, topology.MaxExpertsPerLayer);
    }

    [Fact]
    public void MixedDenseAndSparse_WithSharedGroupingVariableTopKAndDraft_DiscoveredFromTopology()
    {
        List<SparseLayerDescriptor> layers = new List<SparseLayerDescriptor>
        {
            new(0, null, SequenceStateKind.SlidingWindowKv),
            new(1, Layer(experts: 64, topK: 6, topKPrefill: 8, scoring: MoeRouteScoring.SqrtSoftplus, groups: 8, groupsKept: 2,
                shared: new ExpertGroupDescriptor(1, Expert()).Validated(), tokenKind: true,
                program: ExpertProgram.SwigluClamped(10f)), SequenceStateKind.CompressedKv),
            new(2, Layer(experts: 16, topK: 2, scoring: MoeRouteScoring.Sigmoid, routed: new ExpertGroupDescriptor(16, Expert(32),
                new Dictionary<int, ExpertDescriptor> { [3] = Expert(64, DType.F4E2M1) }).Validated()), SequenceStateKind.CompressedKv, IsDraft: true),
        };
        SparseModelTopology topology = new(Hidden, layers, "synthetic");

        SparseCapabilities caps = topology.Capabilities;
        Assert.True(caps.HasDenseLayers);
        Assert.True(caps.HasMixedDenseAndSparseLayers);
        Assert.True(caps.HasSharedExperts);
        Assert.True(caps.VariableExpertsPerLayer);
        Assert.True(caps.VariableTopK);
        Assert.True(caps.PhaseDependentRouting);
        Assert.True(caps.HasGroupRouting);
        Assert.True(caps.HasTokenKindRouting);
        Assert.True(caps.HasDraftLayers);
        Assert.True(caps.HasClampedPrograms);
        Assert.True(caps.HeterogeneousExpertShapes);
        Assert.Contains(DType.F4E2M1, caps.ExpertDTypes);
        Assert.Contains(DType.F32, caps.ExpertDTypes);
        Assert.Equal(
            new HashSet<SequenceStateKind> { SequenceStateKind.SlidingWindowKv, SequenceStateKind.CompressedKv },
            caps.StateKinds.ToHashSet());
    }

    [Fact]
    public void Validation_RejectsInconsistentTopologies()
    {
        Assert.Throws<ArgumentException>(() => new SparseModelTopology(Hidden, Array.Empty<SparseLayerDescriptor>()));
        Assert.Throws<ArgumentException>(() => new SparseModelTopology(Hidden, new[] { new SparseLayerDescriptor(1, null) }));
        Assert.Throws<ArgumentException>(() => new SparseModelTopology(Hidden + 1, new[] { new SparseLayerDescriptor(0, Layer()) }));
        Assert.Throws<ArgumentException>(() => new SparseModelTopology(Hidden, new[]
        {
            new SparseLayerDescriptor(0, Layer(shared: new ExpertGroupDescriptor(1, new ExpertDescriptor(Hidden * 2, 32, DType.F32)))),
        }));
        Assert.Throws<ArgumentException>(() => new RouterDescriptor(8, 9, 9, MoeRouteScoring.Softmax).Validated());
        Assert.Throws<ArgumentException>(() => new RouterDescriptor(10, 2, 2, MoeRouteScoring.Softmax, GroupCount: 4, GroupsKept: 1).Validated());
        Assert.Throws<ArgumentException>(() => new RouterDescriptor(8, 2, 2, MoeRouteScoring.Softmax, GroupCount: 4, GroupsKept: 5).Validated());
        Assert.Throws<ArgumentException>(() => Layer(experts: 8, routed: new ExpertGroupDescriptor(7, Expert()).Validated()));
        Assert.Throws<ArgumentException>(() => new ExpertGroupDescriptor(4, Expert(), new Dictionary<int, ExpertDescriptor> { [4] = Expert() }).Validated());
    }

    [Fact]
    public void RouterLowering_CarriesPhaseTopKAndFlagsAndRefusesWideRouters()
    {
        RouterDescriptor router = new RouterDescriptor(256, 6, 8, MoeRouteScoring.SqrtSoftplus, GroupCount: 8, GroupsKept: 4,
            Renormalize: true, RenormEpsilon: 1e-20f, Scale: 2.5f, LogitDivisor: 2f).Validated();

        MoeRouteArgs decode = router.ToRouteArgs(prefill: false);
        MoeRouteArgs prefill = router.ToRouteArgs(prefill: true);

        Assert.Equal(256, decode.NumExperts);
        Assert.Equal(6, decode.TopK);
        Assert.Equal(8, prefill.TopK);
        Assert.Equal(MoeRouteScoring.SqrtSoftplus, decode.Scoring);
        Assert.True(decode.IsGrouped);
        Assert.Equal(8, decode.GroupCount);
        Assert.Equal(4, decode.GroupsKept);
        Assert.True(decode.Renormalize);
        Assert.Equal(1e-20f, decode.RenormEpsilon);
        Assert.Equal(2.5f, decode.Scale);
        Assert.Equal(2f, decode.LogitDivisor);

        RouterDescriptor wide = new RouterDescriptor(MoeRouteArgs.MaxExperts + 1, 2, 2, MoeRouteScoring.Softmax).Validated();
        Assert.False(wide.CanLowerToBackend);
        Assert.Throws<InvalidOperationException>(() => wide.ToRouteArgs(prefill: false));

        SparseModelTopology topology = Uniform(1, _ => new MoeLayerDescriptor(
            wide, new ExpertGroupDescriptor(wide.NumExperts, Expert()).Validated(), null, false, ExpertProgram.Swiglu).Validated());
        Assert.True(topology.Capabilities.RequiresHostRouting);
    }

    [Fact]
    public void ExpertProgram_ClampMatchesDeepSeekV41Reference()
    {
        const float limit = 10f;
        ExpertProgram program = ExpertProgram.SwigluClamped(limit);

        // Reference from DeepSeekV41MoeExecutor.Forward: gate capped above only, up clamped to +-limit.
        foreach ((float gate, float up) in new[] { (3f, 4f), (12f, -15f), (-30f, 2f), (50f, 50f), (-50f, -50f) })
        {
            float refGate = gate, refUp = up;
            refUp = Math.Clamp(refUp, -limit, limit);
            refGate = MathF.Min(refGate, limit);
            (float g, float u) = program.Clamp(gate, up);
            Assert.Equal(refGate, g);
            Assert.Equal(refUp, u);
        }
        Assert.True(program.IsClamped);
        Assert.False(ExpertProgram.Swiglu.IsClamped);
        Assert.Equal((-30f, 2f), ExpertProgram.Swiglu.Clamp(-30f, 2f));
        Assert.Throws<ArgumentOutOfRangeException>(() => ExpertProgram.SwigluClamped(0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => ExpertProgram.SwigluClamped(-1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => ExpertProgram.SwigluClamped(float.PositiveInfinity));
    }

    [Fact]
    public void ExpertBytes_UseDTypeBlockGeometryAndOverrides()
    {
        Assert.Equal(3L * Hidden * 32 * 4, Expert(32, DType.F32).PayloadBytes);
        // F4E2M1 packs two elements per byte.
        Assert.Equal(3L * Hidden * 32 / 2, Expert(32, DType.F4E2M1).PayloadBytes);

        ExpertGroupDescriptor group = new ExpertGroupDescriptor(4, Expert(32), new Dictionary<int, ExpertDescriptor> { [2] = Expert(64) }).Validated();
        Assert.Equal(64, group.ExpertAt(2).IntermediateSize);
        Assert.Equal(32, group.ExpertAt(0).IntermediateSize);
        Assert.Equal(3 * Expert(32).PayloadBytes + Expert(64).PayloadBytes, group.PayloadBytes);
        Assert.Throws<ArgumentOutOfRangeException>(() => group.ExpertAt(4));
    }

    [Fact]
    public void Fingerprint_IsStableAndSensitiveToRoutingButNotFamilyLabel()
    {
        SparseModelTopology a = Uniform(2, _ => Layer(experts: 8, topK: 2));
        SparseModelTopology same = new(Hidden, a.Layers, family: "relabelled");
        SparseModelTopology differentTopK = Uniform(2, _ => Layer(experts: 8, topK: 3));
        SparseModelTopology differentClamp = Uniform(2, _ => Layer(experts: 8, topK: 2, program: ExpertProgram.SwigluClamped(7f)));

        Assert.Equal(a.Fingerprint, same.Fingerprint);
        Assert.NotEqual(a.Fingerprint, differentTopK.Fingerprint);
        Assert.NotEqual(a.Fingerprint, differentClamp.Fingerprint);
        Assert.Matches("^[0-9a-f]{64}$", a.Fingerprint);
    }
}
