using System.Globalization;
using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Render;

namespace HartsyInference.Engine.Services;

/// <summary>The thumbnail service: reads the file with the document readers (or the point reader for a PLY), draws it with <see cref="MeshRenderer"/> or <see cref="PointCloudRenderer"/> and encodes a PNG.</summary>
internal sealed class MeshThumbnailService : IMeshThumbnailService
{
    /// <inheritdoc/>
    public IReadOnlyList<string> Formats { get; } = [.. MeshDocumentReader.Formats, "ply"];

    /// <inheritdoc/>
    public MeshThumbnail Render(byte[] data, string format, MeshThumbnailRequest? request = null, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentException.ThrowIfNullOrWhiteSpace(format);
        request ??= new MeshThumbnailRequest();
        MeshRenderOptions options = Options(request);
        options.Validate();

        MeshRender render;
        string kind;
        List<string> notes = [];
        if (string.Equals(format, "ply", StringComparison.OrdinalIgnoreCase))
        {
            PointPlyReader.Result points = PointPlyReader.Read(data);
            if (points.Dropped > 0)
                notes.Add($"{points.Dropped} splat(s) fainter than 5 % opacity were left out.");
            render = PointCloudRenderer.Render(points.Cloud, options, cancel);
            kind = "points";
        }
        else
        {
            MeshDocument document = MeshDocumentReader.Read(data, format);
            render = MeshRenderer.Render(document, options, cancel);
            notes.AddRange(render.Notes);
            kind = "mesh";
        }

        TextureImage image = render.Image;
        byte[] png = image.Channels == 4
            ? PngEncoder.Encode(Plane(image, 3, 0), Plane(image, 1, 3), image.Width, image.Height)
            : PngEncoder.Encode(image.Pixels, image.Width, image.Height);
        return new MeshThumbnail { Png = png, Width = image.Width, Height = image.Height, Kind = kind, Notes = notes };
    }

    private static MeshRenderOptions Options(MeshThumbnailRequest request)
    {
        MeshViewpoint named = MeshViewpoint.Named(request.View ?? "iso");
        return new MeshRenderOptions
        {
            Width = request.Width,
            Height = request.Height,
            Viewpoint = new MeshViewpoint(request.Azimuth ?? named.AzimuthDegrees, request.Elevation ?? named.ElevationDegrees),
            FovDegrees = request.FovDegrees,
            Up = (request.Up ?? "y").ToLowerInvariant() switch
            {
                "y" => MeshUpAxis.Y,
                "z" => MeshUpAxis.Z,
                _ => throw new ArgumentException($"Up '{request.Up}' is not y or z.", nameof(request)),
            },
            Background = request.Background is null ? null : Hex(request.Background),
            Samples = request.Samples,
            Textured = request.Textured,
            PointSize = request.PointSize,
        };
    }

    private static byte[] Hex(string text)
    {
        string digits = text.TrimStart('#');
        if (digits.Length != 6 || !int.TryParse(digits, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out int value))
            throw new ArgumentException($"Background '{text}' is not #rrggbb.", nameof(text));
        return [(byte)(value >> 16), (byte)(value >> 8), (byte)value];
    }

    /// <summary>The <paramref name="count"/> channels of an RGBA picture starting at <paramref name="first"/>, interleaved; the colour (first 0, three channels) or the alpha (first 3, one).</summary>
    private static byte[] Plane(TextureImage image, int count, int first)
    {
        int pixels = image.Width * image.Height;
        byte[] plane = new byte[pixels * count];
        for (int i = 0; i < pixels; i++)
        {
            for (int c = 0; c < count; c++)
                plane[i * count + c] = image.Pixels[i * 4 + first + c];
        }
        return plane;
    }
}
