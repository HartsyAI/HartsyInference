using System.Globalization;
using System.Text;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Worlds;

namespace HartsyInference.Engine.Tests.Worlds;

/// <summary>Shared fixtures for world planning tests: a SANA-shaped capability record, a seed image, and camera paths.</summary>
internal static class WorldTestFixtures
{
    /// <summary>A selector that names CUDA without probing a driver, so tests stay CPU-only.</summary>
    public const string CudaSelector = "cuda:0";

    /// <summary>The seed image size the intrinsics tests project against.</summary>
    public const int ImageWidth = 640;

    /// <summary>The seed image height the intrinsics tests project against.</summary>
    public const int ImageHeight = 360;

    /// <summary>The SANA-shaped capabilities: 1+8n frames up to 161, 16 fps, camera required with intrinsics.</summary>
    public static WorldCapabilities SanaShaped(FrameNormalization normalization = FrameNormalization.Reject) => new WorldCapabilities
    {
        Backends = ["cuda"],
        Precisions = ["bf16"],
        DefaultFps = 16,
        Fps = [16],
        DefaultFrames = 81,
        MinFrames = 9,
        Frames = FrameGrid.StepPlusOne(8, normalization, 161),
        CameraModes = [WorldCameraMode.PerFramePose],
        CameraRequired = true,
        Intrinsics = WorldIntrinsicsPolicy.Required,
        Samplers = ["flow-euler"],
        SupportsGuidance = true,
    };

    /// <summary>A catalog-less-looking fixture entry carrying <paramref name="capabilities"/>.</summary>
    public static CatalogEntry FixtureEntry(WorldCapabilities capabilities) => new CatalogEntry
    {
        Id = "fixture-world",
        Modality = Modality.World,
        DisplayName = "Fixture world",
        Architecture = "test",
        Status = ModelStatus.ValidationPending,
        World = capabilities,
    };

    /// <summary>A spec for <paramref name="entry"/>; the local path exists so a load attempt reaches the load hook.</summary>
    public static ModelSpec Spec(CatalogEntry? entry, string requested = "fixture-world") => new ModelSpec
    {
        Requested = requested,
        Modality = Modality.World,
        Catalog = entry,
        LocalPath = typeof(WorldTestFixtures).Assembly.Location,
    };

    /// <summary>A seed image of the given size.</summary>
    public static ImageData Image(int width = ImageWidth, int height = ImageHeight) => new ImageData
    {
        Rgb = new byte[width * height * 3],
        Width = width,
        Height = height,
    };

    /// <summary>Intrinsics that give a 90 degree horizontal and about 59 degree vertical field of view on the seed image.</summary>
    public static WorldIntrinsics InBoundsIntrinsics() => new WorldIntrinsics { Fx = 320, Fy = 320, Cx = 320, Cy = 180 };

    /// <summary><paramref name="count"/> identity camera-to-world poses.</summary>
    public static List<float[]> IdentityPoses(int count)
    {
        List<float[]> poses = new List<float[]>(count);
        for (int i = 0; i < count; i++)
        {
            poses.Add([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
        }
        return poses;
    }

    /// <summary>A JSON camera file body carrying <paramref name="count"/> poses and inline intrinsics.</summary>
    public static string CameraJson(int count)
    {
        StringBuilder sb = new StringBuilder("{\"poses\": [");
        for (int i = 0; i < count; i++)
        {
            if (i > 0)
            {
                sb.Append(", ");
            }
            sb.Append("[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]");
        }
        sb.Append("], \"intrinsics\": {\"fx\": 320, \"fy\": 320, \"cx\": 320, \"cy\": 180}}");
        return sb.ToString();
    }

    /// <summary>Formats a value the way the planner's snapshots do.</summary>
    public static string Invariant(double value) => value.ToString("R", CultureInfo.InvariantCulture);
}
