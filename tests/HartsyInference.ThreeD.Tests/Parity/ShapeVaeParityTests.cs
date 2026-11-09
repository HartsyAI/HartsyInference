using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.CoLa3D;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>TRELLIS.2 shape VAE (f16c32, fp32) encoder and decoder against the Python CPU reference dump. Integration lane: needs
/// <c>HARTSY_PARITY_DIR</c> and the checkpoints under Models.</summary>
/// <remarks>The decoder stops before the mesh step in Python, so only the head fields Python produced are compared. The
/// vertex, intersection and quad-lerp fields are derived from the raw head with the formulas in the reference script.</remarks>
[Trait("Category", "Integration")]
public sealed class ShapeVaeParityTests
{
    private const int EncoderVoxels = 48, EncoderResolution = 32, LatentVoxels = 8, LatentChannels = 32;
    private const int DecoderInputVoxels = 12, DecoderResolution = 16, DecodedVoxels = 1807, HeadChannels = 7;
    private const float VoxelMargin = 0.5f;
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
                float[] raw = ParityCompare.GatherRows(rawFeats.AsReadOnlySpan<float>(), decOrder, HeadChannels);
                results.Add(ParityCompare.Compare("dec_out_vertices", DeriveVertices(raw), refVertices));
                results.Add(ParityCompare.Compare("dec_out_intersected_logits", DeriveIntersected(raw), refIntersected));
                results.Add(ParityCompare.Compare("dec_out_quad_lerp", DeriveQuadLerp(raw), refQuadLerp));
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

    // Reference formulas: vertices = (1 + 2·margin)·sigmoid(raw[0:3]) − margin; intersected logits = raw[3:6]; quad lerp = softplus(raw[6]).
    private static float[] DeriveVertices(float[] raw)
    {
        int rows = raw.Length / HeadChannels;
        float[] vertices = new float[rows * 3];
        for (int r = 0; r < rows; r++)
        {
            for (int a = 0; a < 3; a++)
                vertices[r * 3 + a] = (1f + 2f * VoxelMargin) * Sigmoid(raw[r * HeadChannels + a]) - VoxelMargin;
        }

        return vertices;
    }

    private static float[] DeriveIntersected(float[] raw)
    {
        int rows = raw.Length / HeadChannels;
        float[] logits = new float[rows * 3];
        for (int r = 0; r < rows; r++)
        {
            for (int a = 0; a < 3; a++)
                logits[r * 3 + a] = raw[r * HeadChannels + 3 + a];
        }

        return logits;
    }

    private static float[] DeriveQuadLerp(float[] raw)
    {
        int rows = raw.Length / HeadChannels;
        float[] quad = new float[rows];
        for (int r = 0; r < rows; r++)
            quad[r] = Softplus(raw[r * HeadChannels + 6]);
        return quad;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private static float Softplus(float x) => x > 20f ? x : MathF.Log(1f + MathF.Exp(x));

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
