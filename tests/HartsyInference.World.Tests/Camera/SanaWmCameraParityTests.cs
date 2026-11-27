using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.World.Camera;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.World.Tests.Camera;

/// <summary>SANA-WM camera inputs against the upstream <c>camera.safetensors</c> dumps: run <c>A_demo_pose</c> (the demo
/// pose file) and run <c>B_action161</c> (the action script <c>w-80,dw-40,aw-40</c>), both with the demo intrinsics file.
/// CPU only; needs <c>SANAWM_REFERENCE_RUNS_DIR</c>.</summary>
[Trait("Category", "Integration")]
public sealed class SanaWmCameraParityTests
{
    private const int Width = 1280, Height = 704;
    private const float PoseTolerance = 1e-5f;
    private const float RelativeTolerance = 2e-6f;
    private const float PluckerTolerance = 1e-5f;
    private readonly ITestOutputHelper _output;

    public SanaWmCameraParityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [SanaWmReferenceRunsFact]
    public void DemoPoseRun_CameraTensorsMatchTheDump()
    {
        string run = SanaWmReferenceRunsFactAttribute.Run("A_demo_pose");
        using SafeTensorsLoader camera = Load(Path.Combine(run, "camera.safetensors"));
        float[][] poses = Poses(camera.GetTensor("c2w_input"));
        AssertCameraMatches(run, camera, poses);
    }

    [SanaWmReferenceRunsFact]
    public void ActionRun_ScriptRollsOutTheDumpedPathAndItsCameraTensors()
    {
        string run = SanaWmReferenceRunsFactAttribute.Run("B_action161");
        using JsonDocument index = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(run, "index.json")));
        string script = index.RootElement.GetProperty("camera").GetProperty("action").GetString()!;
        Assert.Equal("w-80,dw-40,aw-40", script);
        using SafeTensorsLoader camera = Load(Path.Combine(run, "camera.safetensors"));

        float[][] poses = SanaWmAction.ToCameraToWorld(script);
        float[][] expected = Poses(camera.GetTensor("c2w_input"));
        Assert.Equal(expected.Length, poses.Length);
        Report("c2w (action rollout)", MaxAbs(expected, poses));
        Assert.True(MaxAbs(expected, poses) <= PoseTolerance);
        AssertCameraMatches(run, camera, poses);
    }

    private void AssertCameraMatches(string run, SafeTensorsLoader camera, float[][] poses)
    {
        using JsonDocument index = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(run, "index.json")));
        JsonElement crop = index.RootElement.GetProperty("camera").GetProperty("crop_meta");
        int sourceWidth = crop.GetProperty("src_size")[0].GetInt32(), sourceHeight = crop.GetProperty("src_size")[1].GetInt32();
        CoverCropMapping mapping = CoverCropMapping.Create(sourceWidth, sourceHeight, Width, Height);
        Assert.Equal(crop.GetProperty("resized_size")[0].GetInt32(), mapping.ResizedWidth);
        Assert.Equal(crop.GetProperty("resized_size")[1].GetInt32(), mapping.ResizedHeight);
        Assert.Equal(crop.GetProperty("crop_offset")[0].GetInt32(), mapping.Left);
        Assert.Equal(crop.GetProperty("crop_offset")[1].GetInt32(), mapping.Top);

        PinholeIntrinsics[] file = ReadIntrinsicsNpy(index.RootElement.GetProperty("args").GetProperty("intrinsics").GetString()!);
        PinholeIntrinsics[] fitted = SanaWmCameraConditioning.FitOverTime(file, poses.Length);
        float fitError = MaxAbs(Rows(camera.GetTensor("intrinsics_src"), 4), Flatten(fitted));
        Report("intrinsics_src (time fit)", fitError);
        Assert.Equal(0f, fitError);

        using SanaWmCameraConditioning conditioning = SanaWmCameraConditioning.Prepare(poses, file, mapping, Width, Height);
        float relError = MaxAbs(Poses(camera.GetTensor("poses_rel")), conditioning.RelativePoses);
        float cropError = MaxRelative(Rows(camera.GetTensor("intrinsics_crop"), 4), Flatten(conditioning.CropIntrinsics));
        float latentError = MaxRelative(Rows(camera.GetTensor("intrinsics_latent"), 4), Flatten(conditioning.LatentIntrinsics));
        float conditionError = MaxAbs(camera.GetTensor("raymap_cond").AsReadOnlySpan<float>(), conditioning.Conditions);
        float pluckerError = MaxAbs(camera.GetTensor("chunk_plucker_cond").AsReadOnlySpan<float>(),
            conditioning.ChunkPlucker.AsReadOnlySpan<float>());
        Report("poses_rel", relError);
        Report("intrinsics_crop (relative)", cropError);
        Report("intrinsics_latent (relative)", latentError);
        Report("camera_conditions", conditionError);
        Report("chunk_plucker", pluckerError);

        Assert.Equal(camera.GetTensor("chunk_plucker_cond").Shape, conditioning.ChunkPlucker.Shape);
        Assert.Equal(21, conditioning.LatentFrames);
        Assert.True(relError <= PoseTolerance, $"poses_rel max abs error {relError}");
        Assert.True(cropError <= RelativeTolerance, $"intrinsics_crop max relative error {cropError}");
        Assert.True(latentError <= RelativeTolerance, $"intrinsics_latent max relative error {latentError}");
        Assert.True(conditionError <= PoseTolerance, $"camera_conditions max abs error {conditionError}");
        Assert.True(pluckerError <= PluckerTolerance, $"chunk_plucker max abs error {pluckerError}");
    }

    private void Report(string name, float error) => _output.WriteLine($"{name}: max error {error:E3}");

    private static SafeTensorsLoader Load(string path)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        return loader;
    }

    private static float[][] Poses(Tensor tensor) => Rows(tensor, 16);

    private static float[][] Rows(Tensor tensor, int width)
    {
        Assert.Equal(DType.F32, tensor.DType);
        ReadOnlySpan<float> values = tensor.AsReadOnlySpan<float>();
        float[][] rows = new float[values.Length / width][];
        for (int i = 0; i < rows.Length; i++)
            rows[i] = values.Slice(i * width, width).ToArray();
        return rows;
    }

    private static float[][] Flatten(IReadOnlyList<PinholeIntrinsics> intrinsics)
    {
        float[][] rows = new float[intrinsics.Count][];
        for (int i = 0; i < rows.Length; i++)
            rows[i] = [intrinsics[i].Fx, intrinsics[i].Fy, intrinsics[i].Cx, intrinsics[i].Cy];
        return rows;
    }

    private static float MaxAbs(IReadOnlyList<float[]> expected, IReadOnlyList<float[]> actual)
    {
        Assert.Equal(expected.Count, actual.Count);
        float max = 0f;
        for (int i = 0; i < expected.Count; i++)
            max = Math.Max(max, MaxAbs(expected[i], actual[i]));
        return max;
    }

    private static float MaxAbs(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        float max = 0f;
        for (int i = 0; i < expected.Length; i++)
            max = Math.Max(max, MathF.Abs(expected[i] - actual[i]));
        return max;
    }

    private static float MaxRelative(IReadOnlyList<float[]> expected, IReadOnlyList<float[]> actual)
    {
        Assert.Equal(expected.Count, actual.Count);
        float max = 0f;
        for (int i = 0; i < expected.Count; i++)
        {
            for (int k = 0; k < expected[i].Length; k++)
                max = Math.Max(max, MathF.Abs(expected[i][k] - actual[i][k]) / Math.Max(MathF.Abs(expected[i][k]), 1e-6f));
        }
        return max;
    }

    /// <summary>Reads a little-endian float32 <c>(F, 3, 3)</c> or <c>(F, 4)</c> intrinsics array, C order.</summary>
    private static PinholeIntrinsics[] ReadIntrinsicsNpy(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        int headerLength = BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(8));
        string header = Encoding.ASCII.GetString(bytes, 10, headerLength);
        Assert.Contains("'descr': '<f4'", header, StringComparison.Ordinal);
        Assert.Contains("'fortran_order': False", header, StringComparison.Ordinal);
        bool matrices = header.Contains(", 3, 3)", StringComparison.Ordinal);
        int stride = matrices ? 9 : 4;
        ReadOnlySpan<byte> data = bytes.AsSpan(10 + headerLength);
        int count = data.Length / (stride * sizeof(float));
        PinholeIntrinsics[] result = new PinholeIntrinsics[count];
        for (int i = 0; i < count; i++)
        {
            int b = i * stride;
            result[i] = matrices
                ? new PinholeIntrinsics(Read(data, b), Read(data, b + 4), Read(data, b + 2), Read(data, b + 5))
                : new PinholeIntrinsics(Read(data, b), Read(data, b + 1), Read(data, b + 2), Read(data, b + 3));
        }
        return result;
    }

    private static float Read(ReadOnlySpan<byte> data, int index) =>
        BinaryPrimitives.ReadSingleLittleEndian(data[(index * sizeof(float))..]);
}
