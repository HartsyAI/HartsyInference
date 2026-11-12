using System.Buffers.Binary;
using System.Text;
using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Render;

namespace HartsyInference.ThreeD.Tests.Render;

/// <summary>The software renderer: placement, depth, views, up axis, textures, colours, the point renderer and the splat reader, all on small synthetic models.</summary>
public sealed class MeshRenderTests
{
    private static readonly MeshRenderOptions Front = new() { Width = 64, Height = 64, Viewpoint = MeshViewpoint.Named("front"), Background = [10, 20, 30], Samples = 1 };

    private static MeshDocument Doc(Mesh mesh, MeshMaterial? material = null) => new()
    {
        Primitives = [new MeshPrimitive { Geometry = mesh, Material = material is null ? -1 : 0 }],
        Materials = material is null ? [] : [material],
    };

    private static Mesh Quad(float z = 0, float size = 1) => new()
    {
        Vertices = [0, 0, z, size, 0, z, size, size, z, 0, size, z],
        Indices = [0, 1, 2, 0, 2, 3],
        Uvs = [0, 1, 1, 1, 1, 0, 0, 0],
    };

    private static Mesh Box(float x, float y, float z) => new()
    {
        Vertices = [0, 0, 0, x, 0, 0, x, y, 0, 0, y, 0, 0, 0, z, x, 0, z, x, y, z, 0, y, z],
        Indices = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5],
    };

    private static (byte R, byte G, byte B) Pixel(TextureImage image, int x, int y)
    {
        int o = (y * image.Width + x) * image.Channels;
        return (image.Pixels[o], image.Pixels[o + 1], image.Pixels[o + 2]);
    }

    private static (int Left, int Right, int Top, int Bottom) Extent(TextureImage image)
    {
        int left = image.Width, right = -1, top = image.Height, bottom = -1;
        for (int y = 0; y < image.Height; y++)
        {
            for (int x = 0; x < image.Width; x++)
            {
                if (image.Pixels[(y * image.Width + x) * 4 + 3] == 0)
                    continue;
                left = Math.Min(left, x);
                right = Math.Max(right, x);
                top = Math.Min(top, y);
                bottom = Math.Max(bottom, y);
            }
        }
        return (left, right, top, bottom);
    }

    [Fact]
    public void AModelIsDrawnInTheMiddleOnTheBackgroundAndKeepsItsMarginAtTheEdges()
    {
        MeshRender render = MeshRenderer.Render(Doc(Box(1, 1, 1)), Front);

        Assert.Equal((10, 20, 30), Pixel(render.Image, 0, 0));
        Assert.Equal((10, 20, 30), Pixel(render.Image, 63, 63));
        (byte r, byte g, byte b) = Pixel(render.Image, 32, 32);
        Assert.NotEqual((10, 20, 30), (r, g, b));
        Assert.Equal(3, render.Image.Channels);
    }

    [Fact]
    public void AnImageWithoutABackgroundIsRgbaWithCoverageAsAlpha()
    {
        MeshRender render = MeshRenderer.Render(Doc(Box(1, 1, 1)), Front with { Background = null, Samples = 2 });

        Assert.Equal(4, render.Image.Channels);
        Assert.Equal(0, render.Image.Pixels[3]);
        Assert.Equal(255, render.Image.Pixels[(32 * 64 + 32) * 4 + 3]);
        Assert.Equal((64, 64), (render.Image.Width, render.Image.Height));
    }

    [Fact]
    public void ANearerSurfaceHidesAFartherOneAndAViewFromTheOtherSideSeesTheOther()
    {
        MeshMaterial red = new() { BaseColorFactor = [1, 0, 0, 1] }, blue = new() { BaseColorFactor = [0, 0, 1, 1] };
        MeshDocument document = new()
        {
            Primitives = [new MeshPrimitive { Geometry = Quad(0), Material = 0 }, new MeshPrimitive { Geometry = Quad(1), Material = 1 }],
            Materials = [red, blue],
        };

        (byte r, byte g, byte b) frontPixel = Pixel(MeshRenderer.Render(document, Front).Image, 32, 32);
        (byte r, byte g, byte b) backPixel = Pixel(MeshRenderer.Render(document, Front with { Viewpoint = MeshViewpoint.Named("back") }).Image, 32, 32);

        Assert.True(frontPixel.b > frontPixel.r, "The quad at z=1 is nearer the front camera.");
        Assert.True(backPixel.r > backPixel.b, "The quad at z=0 is nearer the back camera.");
    }

    [Fact]
    public void ATexturedQuadShowsEachTexelInItsCornerWithTheFileOriginAtTheTopLeft()
    {
        MeshImage texture = new("image/png", PngTextureEncoder.Encode(new TextureImage(2, 2, 3,
            [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])));
        MeshDocument document = Doc(Quad(), new MeshMaterial { BaseColor = texture });

        TextureImage image = MeshRenderer.Render(document, Front with { FovDegrees = 30 }).Image;

        (byte r, byte g, byte b) topLeft = Pixel(image, 24, 24), topRight = Pixel(image, 40, 24), bottomLeft = Pixel(image, 24, 40);
        Assert.True(topLeft.r > topLeft.g && topLeft.r > topLeft.b, "top-left texel is red");
        Assert.True(topRight.g > topRight.r && topRight.g > topRight.b, "top-right texel is green");
        Assert.True(bottomLeft.b > bottomLeft.r && bottomLeft.b > bottomLeft.g, "bottom-left texel is blue");
    }

    [Fact]
    public void WithoutTexturesTheMaterialColourIsDrawnAndAJpegIsNotedNotFailed()
    {
        MeshImage jpeg = new("image/jpeg", [0xFF, 0xD8, 0xFF]);
        MeshDocument document = Doc(Quad(), new MeshMaterial { BaseColor = jpeg, BaseColorFactor = [0, 1, 0, 1] });

        MeshRender render = MeshRenderer.Render(document, Front);

        (byte r, byte g, byte b) = Pixel(render.Image, 32, 32);
        Assert.True(g > r && g > b);
        Assert.Contains(render.Notes, n => n.Contains("JPEG"));
        Assert.Empty(MeshRenderer.Render(document, Front with { Textured = false }).Notes);
    }

    [Fact]
    public void VertexColoursTintTheSurface()
    {
        Mesh quad = Quad();
        quad.VertexColors = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0];

        (byte r, byte g, byte b) = Pixel(MeshRenderer.Render(Doc(quad), Front).Image, 32, 32);

        Assert.True(r > g + 40 && r > b + 40);
    }

    [Fact]
    public void AZUpModelStandsUpOnlyWhenItIsToldItIsZUp()
    {
        MeshDocument tall = Doc(Box(1, 1, 4));
        MeshRenderOptions options = Front with { Background = null, Samples = 1 };

        (int l, int r, int t, int b) zUp = Extent(MeshRenderer.Render(tall, options with { Up = MeshUpAxis.Z }).Image);
        (int l, int r, int t, int b) yUp = Extent(MeshRenderer.Render(tall, options with { Up = MeshUpAxis.Y }).Image);

        Assert.True(zUp.b - zUp.t > zUp.r - zUp.l, "Z-up: the long axis is vertical.");
        Assert.True(yUp.r - yUp.l <= yUp.b - yUp.t + 2 && yUp.b - yUp.t < zUp.b - zUp.t, "Y-up: the long axis points at the camera, so it is shorter on screen.");
    }

    [Fact]
    public void TheViewsLookFromDifferentSidesAndAnUnknownNameNamesTheValidOnes()
    {
        MeshDocument document = Doc(Box(1, 1, 3));
        MeshRenderOptions options = Front with { Background = null };

        (int, int, int, int) front = Extent(MeshRenderer.Render(document, options).Image);
        (int, int, int, int) top = Extent(MeshRenderer.Render(document, options with { Viewpoint = MeshViewpoint.Named("top") }).Image);
        (int, int, int, int) iso = Extent(MeshRenderer.Render(document, options with { Viewpoint = MeshViewpoint.Named("iso") }).Image);

        Assert.NotEqual(front, top);
        Assert.NotEqual(front, iso);
        ArgumentException error = Assert.Throws<ArgumentException>(() => MeshViewpoint.Named("sideways"));
        Assert.Contains("front, back, left, right, top, bottom, iso", error.Message);
    }

    [Theory]
    [InlineData(8, 64, 2, 30f)]
    [InlineData(64, 4096, 2, 30f)]
    [InlineData(64, 64, 0, 30f)]
    [InlineData(64, 64, 5, 30f)]
    [InlineData(64, 64, 2, 5f)]
    public void OutOfRangeOptionsAreRefused(int width, int height, int samples, float fov)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => MeshRenderer.Render(Doc(Quad()), new MeshRenderOptions { Width = width, Height = height, Samples = samples, FovDegrees = fov }));
    }

    [Fact]
    public void ThePointRendererDrawsDotsWithTheirColoursAndHidesTheFarOnes()
    {
        PointCloud cloud = new()
        {
            Positions = [-1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1],
            Colors = [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
        };

        TextureImage image = PointCloudRenderer.Render(cloud, Front with { PointSize = 5, Background = [0, 0, 0] }).Image;

        (byte r, byte g, byte b) centre = Pixel(image, 32, 32);
        Assert.True(centre.b > 200 && centre.r < 50, "The point at z=1 is nearest the front camera and hides the one behind it.");
        Assert.True(Enumerable.Range(0, 64).Any(x => Pixel(image, x, 32).R > 200), "The red point is drawn.");
    }

    [Fact]
    public void ASplatPlyIsReadAsPointsWithColoursFromItsDcTermsAndFaintSplatsAreDropped()
    {
        byte[] ply = SplatPly([(0f, 0f, 0f, 1f, 10f), (1f, 0f, 0f, -1f, -10f), (0f, 1f, 0f, 0f, 3f)]);

        PointPlyReader.Result result = PointPlyReader.Read(ply);

        Assert.True(result.IsSplat);
        Assert.Equal(2, result.Cloud.Count);
        Assert.Equal(1, result.Dropped);
        Assert.Equal(0.5f + 0.28209479f, result.Cloud.Colors![0], 4);
        Assert.Equal(0.5f, result.Cloud.Colors[3], 4);
    }

    [Fact]
    public void AnAsciiPlyWithByteColoursIsReadAndABadFileIsADomainError()
    {
        byte[] ascii = Encoding.ASCII.GetBytes("ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n0 0 0 255 0 0\n1 1 1 0 0 255\n");

        PointPlyReader.Result result = PointPlyReader.Read(ascii);

        Assert.False(result.IsSplat);
        Assert.Equal([1f, 0f, 0f, 0f, 0f, 1f], result.Cloud.Colors!);
        Assert.Throws<HartsyInferenceException>(() => PointPlyReader.Read([1, 2, 3]));
        Assert.Throws<HartsyInferenceException>(() => PointPlyReader.Read(ascii[..^10]));
    }

    private static byte[] SplatPly((float X, float Y, float Z, float Dc, float Opacity)[] splats)
    {
        string header = $"ply\nformat binary_little_endian 1.0\nelement vertex {splats.Length}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nproperty float opacity\nproperty float scale_0\nend_header\n";
        byte[] head = Encoding.ASCII.GetBytes(header);
        byte[] data = new byte[head.Length + splats.Length * 9 * 4];
        head.CopyTo(data, 0);
        int o = head.Length;
        foreach ((float x, float y, float z, float dc, float opacity) in splats)
        {
            foreach (float v in new[] { x, y, z, 0f, dc, 0f, 0f, opacity, 0f })
            {
                BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(o), v);
                o += 4;
            }
        }
        return data;
    }
}
