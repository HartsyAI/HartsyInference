using System.Buffers.Binary;
using System.Numerics;
using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.Vision.Codec;

namespace HartsyInference.ThreeD.Render;

/// <summary>Draws a <see cref="MeshDocument"/> into a picture with a software rasteriser: a perspective camera on an orbit, a depth buffer, a headlight, and supersampling for smooth edges. It reads the document, not the model that made it, so it draws every 3D model's mesh the same way. Both sides of every triangle are drawn and lit.</summary>
public static class MeshRenderer
{
    /// <summary>Draws <paramref name="document"/>.</summary>
    /// <returns>The picture (RGBA, or RGB with a background) and notes about what was not drawn as the file describes it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">An option is out of range.</exception>
    public static MeshRender Render(MeshDocument document, MeshRenderOptions options, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(document);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        List<string> notes = [];

        Vector3 min = new(float.PositiveInfinity), max = new(float.NegativeInfinity);
        List<(MeshPrimitive Primitive, Prepared Geometry)> prepared = [];
        foreach (MeshPrimitive primitive in document.Primitives)
        {
            if (primitive.Geometry.VertexCount == 0)
                continue;
            Prepared p = Prepare(primitive.Geometry, options.Up);
            prepared.Add((primitive, p));
            min = Vector3.Min(min, p.Min);
            max = Vector3.Max(max, p.Max);
        }
        if (prepared.Count == 0)
            throw new HartsyInferenceException("The model has no geometry to draw.");

        SampleBuffer buffer = new(options);
        OrbitCamera camera = new(min, max, options, buffer.Width, buffer.Height);
        Dictionary<MeshImage, Texture?> textures = new(ReferenceEqualityComparer.Instance);
        bool jpegSkipped = false, undecodable = false;
        foreach ((MeshPrimitive primitive, Prepared geometry) in prepared)
        {
            cancel.ThrowIfCancellationRequested();
            MeshMaterial? material = (uint)primitive.Material < (uint)document.Materials.Count ? document.Materials[primitive.Material] : null;
            Texture? texture = null;
            if (options.Textured && material?.BaseColor is { } image && primitive.Geometry.Uvs is not null)
            {
                if (!textures.TryGetValue(image, out texture))
                {
                    texture = image.MimeType == "image/png" ? Texture.TryDecode(image.Data) : null;
                    textures[image] = texture;
                }
                if (texture is null)
                {
                    jpegSkipped |= image.MimeType != "image/png";
                    undecodable |= image.MimeType == "image/png";
                }
            }
            Draw(geometry, material, texture, camera, buffer, cancel);
        }
        if (jpegSkipped)
            notes.Add("A base-colour texture is a JPEG, which the renderer does not decode, so that material is drawn with its colour only.");
        if (undecodable)
            notes.Add($"A base-colour PNG texture could not be decoded (it is corrupt or larger than {Texture.MaxPixels} pixels), so that material is drawn with its colour only.");
        return new MeshRender(buffer.Resolve(options), notes);
    }

    /// <summary>The geometry turned to Y-up, with its bounds and the vertex colours in display (sRGB) values.</summary>
    private static Prepared Prepare(Mesh mesh, MeshUpAxis up)
    {
        int n = mesh.VertexCount;
        Vector3[] positions = new Vector3[n];
        Vector3 min = new(float.PositiveInfinity), max = new(float.NegativeInfinity);
        for (int i = 0; i < n; i++)
        {
            positions[i] = OrbitCamera.ToYUp(new Vector3(mesh.Vertices[i * 3], mesh.Vertices[i * 3 + 1], mesh.Vertices[i * 3 + 2]), up);
            min = Vector3.Min(min, positions[i]);
            max = Vector3.Max(max, positions[i]);
        }
        Vector3[]? normals = null;
        if (mesh.Normals is { } source)
        {
            normals = new Vector3[n];
            for (int i = 0; i < n; i++)
                normals[i] = OrbitCamera.ToYUp(new Vector3(source[i * 3], source[i * 3 + 1], source[i * 3 + 2]), up);
        }
        Vector3[]? colours = null;
        if (mesh.VertexColors is { } c)
        {
            colours = new Vector3[n];
            for (int i = 0; i < n; i++)
                colours[i] = new Vector3(ToDisplay(c[i * 3]), ToDisplay(c[i * 3 + 1]), ToDisplay(c[i * 3 + 2]));
        }
        return new Prepared(mesh, positions, normals, colours, min, max);
    }

    private static void Draw(Prepared p, MeshMaterial? material, Texture? texture, OrbitCamera camera, SampleBuffer buffer, CancellationToken cancel)
    {
        // Colour space: texels are sRGB and the factors and vertex colours are converted to the same display space before they are
        // multiplied, so all three terms are in one space; change them together or not at all.
        float[] f = material?.BaseColorFactor ?? [1f, 1f, 1f, 1f];
        Vector3 factor = new(ToDisplay(f[0]), ToDisplay(f[1]), ToDisplay(f[2]));
        float alphaFactor = f[3];
        bool mask = material?.AlphaMode == MeshAlphaMode.Mask;
        float cutoff = material?.AlphaCutoff ?? 0.5f;
        float[]? uvs = p.Mesh.Uvs;
        int[] indices = p.Mesh.Indices;

        (float X, float Y, float W)[] screen = new (float, float, float)[p.Positions.Length];
        for (int i = 0; i < screen.Length; i++)
            screen[i] = camera.Project(p.Positions[i]);

        for (int t = 0; t < indices.Length; t += 3)
        {
            if ((t / 3 & 0x1FFF) == 0)
                cancel.ThrowIfCancellationRequested();
            int a = indices[t], b = indices[t + 1], c = indices[t + 2];
            (float x0, float y0, float w0) = screen[a];
            (float x1, float y1, float w1) = screen[b];
            (float x2, float y2, float w2) = screen[c];
            if (w0 <= 0 || w1 <= 0 || w2 <= 0)
                continue;
            float area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
            if (MathF.Abs(area) < 1e-12f)
                continue;
            int minX = Math.Max(0, (int)MathF.Floor(MathF.Min(x0, MathF.Min(x1, x2))));
            int maxX = Math.Min(buffer.Width - 1, (int)MathF.Ceiling(MathF.Max(x0, MathF.Max(x1, x2))));
            int minY = Math.Max(0, (int)MathF.Floor(MathF.Min(y0, MathF.Min(y1, y2))));
            int maxY = Math.Min(buffer.Height - 1, (int)MathF.Ceiling(MathF.Max(y0, MathF.Max(y1, y2))));
            if (minX > maxX || minY > maxY)
                continue;

            Vector3 va = p.Positions[a], vb = p.Positions[b], vc = p.Positions[c];
            Vector3 face = Vector3.Cross(vb - va, vc - va);
            float faceLength = face.Length();
            if (faceLength <= 0)
                continue;
            face /= faceLength;
            // Which side of the triangle the camera sees; the shading normal is turned towards it.
            float side = Vector3.Dot(face, camera.Eye - va) >= 0 ? 1f : -1f;
            float invArea = 1f / area;
            float iw0 = 1f / w0, iw1 = 1f / w1, iw2 = 1f / w2;

            for (int y = minY; y <= maxY; y++)
            {
                float py = y + 0.5f;
                for (int x = minX; x <= maxX; x++)
                {
                    float px = x + 0.5f;
                    float l0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) * invArea;
                    float l1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) * invArea;
                    float l2 = 1f - l0 - l1;
                    if (l0 < -1e-5f || l1 < -1e-5f || l2 < -1e-5f)
                        continue;
                    float pw0 = l0 * iw0, pw1 = l1 * iw1, pw2 = l2 * iw2;
                    float depth = 1f / (pw0 + pw1 + pw2);
                    if (!buffer.IsNearer(x, y, depth))
                        continue;
                    float b0 = pw0 * depth, b1 = pw1 * depth, b2 = pw2 * depth;

                    Vector3 colour = factor;
                    if (p.Colours is { } vertexColours)
                        colour *= vertexColours[a] * b0 + vertexColours[b] * b1 + vertexColours[c] * b2;
                    float alpha = alphaFactor;
                    if (texture is not null && uvs is not null)
                    {
                        float u = uvs[a * 2] * b0 + uvs[b * 2] * b1 + uvs[c * 2] * b2;
                        float v = uvs[a * 2 + 1] * b0 + uvs[b * 2 + 1] * b1 + uvs[c * 2 + 1] * b2;
                        (Vector3 texel, float texelAlpha) = texture.Sample(u, v);
                        colour *= texel;
                        alpha *= texelAlpha;
                    }
                    if (mask && alpha < cutoff)
                        continue;

                    Vector3 normal = face;
                    if (p.Normals is { } vertexNormals)
                    {
                        Vector3 blended = vertexNormals[a] * b0 + vertexNormals[b] * b1 + vertexNormals[c] * b2;
                        if (blended.LengthSquared() > 1e-12f)
                            normal = Vector3.Normalize(blended);
                        // Vertex normals can point away from the side the camera sees (a mirrored or two-sided mesh): keep them with the face.
                        if (Vector3.Dot(normal, face) < 0)
                            normal = -normal;
                    }
                    normal *= side;
                    float shade = 0.35f + 0.65f * MathF.Max(0f, Vector3.Dot(normal, camera.Light));
                    colour *= shade;
                    buffer.TryWrite(x, y, depth, ToByte(colour.X), ToByte(colour.Y), ToByte(colour.Z));
                }
            }
        }
    }

    private static float ToDisplay(float linear)
    {
        float c = Math.Clamp(linear, 0f, 1f);
        return c <= 0.0031308f ? 12.92f * c : 1.055f * MathF.Pow(c, 1f / 2.4f) - 0.055f;
    }

    private static byte ToByte(float v) => (byte)MathF.Round(Math.Clamp(v, 0f, 1f) * 255f);

    private sealed record Prepared(Mesh Mesh, Vector3[] Positions, Vector3[]? Normals, Vector3[]? Colours, Vector3 Min, Vector3 Max);

    /// <summary>A decoded PNG base-colour image sampled with repeat wrapping and nearest filtering.</summary>
    private sealed class Texture
    {
        private readonly byte[] _rgb;
        private readonly byte[]? _alpha;
        private readonly int _width, _height;

        private Texture(DecodedPng png)
        {
            _rgb = png.Rgb;
            _alpha = png.Alpha;
            _width = png.Width;
            _height = png.Height;
        }

        /// <summary>The most pixels a texture may have (8192 x 8192); the decoder is not run on a larger one.</summary>
        public const long MaxPixels = 8192L * 8192;

        public static Texture? TryDecode(byte[] data)
        {
            // The PNG header names the size before any of the image is inflated: width and height are the first IHDR fields.
            if (data.Length < 24 || (long)BinaryPrimitives.ReadUInt32BigEndian(data.AsSpan(16)) * BinaryPrimitives.ReadUInt32BigEndian(data.AsSpan(20)) > MaxPixels)
                return null;
            try
            {
                return new Texture(PngDecoder.DecodeWithAlpha(data));
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                return null;
            }
        }

        public (Vector3 Colour, float Alpha) Sample(float u, float v)
        {
            float fu = u - MathF.Floor(u), fv = v - MathF.Floor(v);
            // Clamped so a NaN coordinate from a future reader cannot index outside the image.
            int x = Math.Clamp((int)(fu * _width), 0, _width - 1), y = Math.Clamp((int)(fv * _height), 0, _height - 1);
            int i = y * _width + x;
            return (new Vector3(_rgb[i * 3], _rgb[i * 3 + 1], _rgb[i * 3 + 2]) / 255f, _alpha is null ? 1f : _alpha[i] / 255f);
        }
    }
}
