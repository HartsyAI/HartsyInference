using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Document;

namespace HartsyInference.ThreeD.Io;

public static partial class GlbReader
{
    private const string DataUriPrefix = "data:";

    /// <summary>Reads the materials with their embedded images. An image that is not embedded, or is neither PNG nor JPEG, leaves its slot empty and is noted.</summary>
    private static List<MeshMaterial> ReadMaterials(JsonElement root, byte[] bin, List<string> notes)
    {
        List<MeshMaterial> materials = [];
        if (!TryArray(root, "materials", out JsonElement list))
            return materials;
        Dictionary<int, MeshImage?> images = [];
        bool notedUvSet = false;
        foreach (JsonElement material in list.EnumerateArray())
        {
            JsonElement pbr = TryMember(material, "pbrMetallicRoughness", out JsonElement found) ? found : default;
            MeshImage? Slot(JsonElement owner, string name)
            {
                if (!TryMember(owner, name, out JsonElement info))
                    return null;
                if (IntMember(info, "texCoord", 0) != 0)
                {
                    if (!notedUvSet)
                        notes.Add("A texture reads UV set 1 or higher; only UV set 0 is carried, so that texture was left out.");
                    notedUvSet = true;
                    return null;
                }
                int texture = NaturalMember(info, "index");
                JsonElement textureEntry = Item(RequiredArray(root, "textures"), texture, "texture");
                if (!TryMember(textureEntry, "source", out JsonElement source))
                {
                    notes.Add($"Texture {texture} has no image source (only PNG and JPEG images are read); it was left out.");
                    return null;
                }
                int image = Natural(source, "image index");
                if (!images.TryGetValue(image, out MeshImage? loaded))
                {
                    loaded = ReadImage(root, bin, image, notes);
                    images[image] = loaded;
                }
                return loaded;
            }

            float[] baseColor = FloatsMember(pbr, "baseColorFactor", 4) ?? [1f, 1f, 1f, 1f];
            string alphaMode = TryMember(material, "alphaMode", out JsonElement mode) && mode.ValueKind == JsonValueKind.String ? mode.GetString()! : "OPAQUE";
            materials.Add(new MeshMaterial
            {
                Name = TryMember(material, "name", out JsonElement name) && name.ValueKind == JsonValueKind.String ? name.GetString()! : "",
                BaseColorFactor = baseColor,
                BaseColor = Slot(pbr, "baseColorTexture"),
                Metallic = FloatMember(pbr, "metallicFactor", 1f),
                Roughness = FloatMember(pbr, "roughnessFactor", 1f),
                MetallicRoughness = Slot(pbr, "metallicRoughnessTexture"),
                Normal = Slot(material, "normalTexture"),
                Occlusion = Slot(material, "occlusionTexture"),
                EmissiveFactor = FloatsMember(material, "emissiveFactor", 3) ?? [0f, 0f, 0f],
                Emissive = Slot(material, "emissiveTexture"),
                AlphaMode = alphaMode switch { "MASK" => MeshAlphaMode.Mask, "BLEND" => MeshAlphaMode.Blend, _ => MeshAlphaMode.Opaque },
                AlphaCutoff = FloatMember(material, "alphaCutoff", 0.5f),
                DoubleSided = TryMember(material, "doubleSided", out JsonElement sided) && sided.ValueKind == JsonValueKind.True,
            });
        }
        return materials;
    }

    private static MeshImage? ReadImage(JsonElement root, byte[] bin, int index, List<string> notes)
    {
        JsonElement image = Item(RequiredArray(root, "images"), index, "image");
        string? mime = TryMember(image, "mimeType", out JsonElement m) && m.ValueKind == JsonValueKind.String ? m.GetString() : null;
        byte[] data;
        if (TryMember(image, "bufferView", out JsonElement viewIndex))
        {
            JsonElement view = Item(RequiredArray(root, "bufferViews"), Natural(viewIndex, "buffer view index"), "bufferView");
            long offset = NaturalMember(view, "byteOffset", 0);
            long length = NaturalMember(view, "byteLength");
            if (NaturalMember(view, "buffer") != 0 || offset + length > bin.Length)
            {
                notes.Add($"Image {index} reads outside the GLB binary chunk and was left out.");
                return null;
            }
            data = bin.AsSpan((int)offset, (int)length).ToArray();
        }
        else if (TryMember(image, "uri", out JsonElement uri) && uri.ValueKind == JsonValueKind.String && uri.GetString()!.StartsWith(DataUriPrefix, StringComparison.Ordinal))
        {
            string text = uri.GetString()!;
            int comma = text.IndexOf(',');
            if (comma < 0 || !text[..comma].EndsWith(";base64", StringComparison.Ordinal))
            {
                notes.Add($"Image {index} is a data URI that is not base64 and was left out.");
                return null;
            }
            mime ??= text[DataUriPrefix.Length..text.IndexOf(';')];
            try
            {
                data = Convert.FromBase64String(text[(comma + 1)..]);
            }
            catch (FormatException ex)
            {
                throw new HartsyInferenceException($"GLB image {index} holds invalid base64.", ex);
            }
        }
        else
        {
            notes.Add($"Image {index} is not embedded in the file (an external file is not read) and was left out.");
            return null;
        }
        mime ??= data.Length > 3 && data[0] == 0xFF && data[1] == 0xD8 ? "image/jpeg" : "image/png";
        if (mime is not ("image/png" or "image/jpeg"))
        {
            notes.Add($"Image {index} is {mime}; only PNG and JPEG are carried, so it was left out.");
            return null;
        }
        return new MeshImage(mime, data);
    }

    private static float FloatMember(JsonElement obj, string name, float fallback) =>
        TryMember(obj, name, out JsonElement value) && value.TryGetSingle(out float number) && float.IsFinite(number) ? number : fallback;
}
