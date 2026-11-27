using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.CoLa3D;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>TRELLIS.2 shape VAE (f16c32, fp32) encoder and decoder against the Python CPU reference dump. Integration lane: needs
/// <c>HARTSY_PARITY_DIR</c> and the checkpoints under Models.</summary>
/// <remarks>The decoder stops before the mesh step in Python. The vertex, intersection-logit and quad-lerp fields are read from
/// the production <see cref="ShapeVaeHead.Split"/> output, not recomputed here.</remarks>
[Trait("Category", "Integration")]
public sealed class ShapeVaeParityTests
{
    private const int EncoderVoxels = 48, EncoderResolution = 32, LatentVoxels = 8, LatentChannels = 32;
    private const int DecoderInputVoxels = 12, DecoderResolution = 16, DecodedVoxels = 1807, HeadChannels = 7;
    private readonly ITestOutputHelper _output;

    public ShapeVaeParityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void F16C32Fp32_MatchesPythonCpuReferenceStageByStage()
    {
        if (!ParityFixtures.TryResolve([ParityFixtures.ShapeVaeFixture], out string parityDir, out string modelRoot, out string reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }

        NpzArchive fixture = NpzArchive.Open(Path.Combine(parityDir, ParityFixtures.ShapeVaeFixture));
        int[] encCoords = fixture.Get("enc_in_coords").RequireShape(EncoderVoxels, 4).ToInt32();
        float[] encVertices = fixture.Get("enc_in_vertices").RequireShape(EncoderVoxels, 3).ToFloat32();
        float[] encIntersected = ToFloats(fixture.Get("enc_in_intersected").RequireShape(EncoderVoxels, 3).ToBool());
        int[] refLatentCoords = fixture.Get("enc_out_latent_coords").RequireShape(LatentVoxels, 4).ToInt32();
        float[] refLatentMean = fixture.Get("enc_out_latent_mean").RequireShape(LatentVoxels, LatentChannels).ToFloat32();
        float[] refLatentLogVar = fixture.Get("enc_out_latent_logvar").RequireShape(LatentVoxels, LatentChannels).ToFloat32();
        int[] decInCoords = fixture.Get("dec_in_coords").RequireShape(DecoderInputVoxels, 4).ToInt32();
        float[] decInLatent = fixture.Get("dec_in_latent_feats").RequireShape(DecoderInputVoxels, LatentChannels).ToFloat32();
        int[] refDecCoords = fixture.Get("dec_out_coords").RequireShape(DecodedVoxels, 4).ToInt32();
        float[] refRaw = fixture.Get("dec_out_raw_feats").RequireShape(DecodedVoxels, HeadChannels).ToFloat32();
        float[] refVertices = fixture.Get("dec_out_vertices").RequireShape(DecodedVoxels, 3).ToFloat32();
        float[] refIntersected = fixture.Get("dec_out_intersected_logits").RequireShape(DecodedVoxels, 3).ToFloat32();
        float[] refQuadLerp = fixture.Get("dec_out_quad_lerp").RequireShape(DecodedVoxels, 1).ToFloat32();

        ShapeVaeConfig shape = ShapeVaeConfig.NextDcF16C32;
        using CoLa3DWeights weights = CoLa3DCheckpointConverter.Load(modelRoot);
        using CpuBackend backend = new();
        using ShapeVaeEncoder encoder = new(shape.Encoder, weights.ShapeEncoder);
        using ShapeVaeDecoder decoder = new(shape.Decoder, weights.ShapeDecoder);

        Tensor vertexFeats = ToTensor(encVertices, [EncoderVoxels, 3]);
        Tensor intersectedFeats = ToTensor(encIntersected, [EncoderVoxels, 3]);
        Tensor latentFeats = ToTensor(decInLatent, [DecoderInputVoxels, LatentChannels]);
        Tensor? meanFeats = null, logVarFeats = null, rawFeats = null;
        try
        {
            SparseTensor vertices = new(vertexFeats, encCoords, EncoderResolution);
            SparseTensor intersected = new(intersectedFeats, encCoords, EncoderResolution);
            (SparseTensor mean, Tensor logVar) = encoder.EncodeMoments(backend, vertices, intersected);
            meanFeats = mean.Feats;
            logVarFeats = logVar;

            List<ParityResult> results = [];
            ParityResult latentCoords = ParityCompare.CompareCoords("enc_out_latent_coords", mean.Coords, refLatentCoords, out int[] latentOrder);
            results.Add(latentCoords);
            results.Add(CompareOrdered("enc_out_latent_mean", latentCoords, meanFeats, latentOrder, LatentChannels, refLatentMean));
            results.Add(CompareOrdered("enc_out_latent_logvar", latentCoords, logVarFeats, latentOrder, LatentChannels, refLatentLogVar));

            SparseTensor latent = new(latentFeats, decInCoords, DecoderResolution);
            SparseTensor head = decoder.Decode(backend, latent);
            rawFeats = head.Feats;
            ParityResult decCoords = ParityCompare.CompareCoords("dec_out_coords", head.Coords, refDecCoords, out int[] decOrder);
            results.Add(decCoords);
            results.Add(CompareOrdered("dec_out_raw_feats", decCoords, rawFeats, decOrder, HeadChannels, refRaw));

            if (decCoords.Passed)
            {
                DualGridField field = ShapeVaeHead.Split(head);
                float[] logits = field.IntersectionLogits
                    ?? throw new InvalidOperationException("ShapeVaeHead.Split did not return intersection logits.");
                float[] absolute = ParityCompare.GatherRows(field.Voxels.DualVertices, decOrder, 3);
                float[] offsets = new float[absolute.Length];
                for (int row = 0; row < DecodedVoxels; row++)
                {
                    for (int axis = 0; axis < 3; axis++)
                        offsets[row * 3 + axis] = absolute[row * 3 + axis] - refDecCoords[row * 4 + 1 + axis];
                }

                results.Add(ParityCompare.Compare("dec_out_vertices", offsets, refVertices));
                results.Add(ParityCompare.Compare("dec_out_intersected_logits", ParityCompare.GatherRows(logits, decOrder, 3), refIntersected));
                results.Add(ParityCompare.Compare("dec_out_quad_lerp", ParityCompare.GatherRows(field.SplitWeight, decOrder, 1), refQuadLerp));
            }
            else
            {
                results.Add(ParityResult.Failed("dec_out_vertices", "skipped: decoder voxel sets differ"));
                results.Add(ParityResult.Failed("dec_out_intersected_logits", "skipped: decoder voxel sets differ"));
                results.Add(ParityResult.Failed("dec_out_quad_lerp", "skipped: decoder voxel sets differ"));
            }

            ParityCompare.Finish(results, _output);
        }
        finally
        {
            meanFeats?.Dispose();
            logVarFeats?.Dispose();
            rawFeats?.Dispose();
            vertexFeats.Dispose();
            intersectedFeats.Dispose();
            latentFeats.Dispose();
        }
    }

    /// <summary>Compares a feature array after gathering its rows into reference order; skipped when the coordinate stage failed.</summary>
    private static ParityResult CompareOrdered(string name, ParityResult coords, Tensor feats, int[] order, int width, float[] reference)
    {
        if (!coords.Passed)
            return ParityResult.Failed(name, "skipped: voxel sets differ");
        return ParityCompare.Compare(name, ParityCompare.GatherRows(feats.AsReadOnlySpan<float>(), order, width), reference);
    }

    private static float[] ToFloats(bool[] flags)
    {
        float[] values = new float[flags.Length];
        for (int i = 0; i < flags.Length; i++)
            values[i] = flags[i] ? 1f : 0f;
        return values;
    }

    private static Tensor ToTensor(float[] values, long[] shape)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32);
        values.AsSpan().CopyTo(tensor.AsSpan<float>());
        return tensor;
    }
}
