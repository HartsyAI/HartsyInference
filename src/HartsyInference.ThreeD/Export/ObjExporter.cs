using System.Globalization;
using System.Text;
using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Export;

/// <summary>Wavefront OBJ with an MTL beside it and the material images as files: positions, UVs (OBJ's origin is bottom-left, so V is flipped), normals, one group per primitive and one material per glTF material. Base colour, normal and emissive maps have MTL slots; metallic-roughness and occlusion maps do not and are left out with a note.</summary>
public sealed class ObjExporter : IMeshExporter
{
    /// <inheritdoc/>
    public string Id => "obj";

    /// <inheritdoc/>
    public MeshExport Export(MeshDocument document, string name)
    {
        ArgumentNullException.ThrowIfNull(document);
        string stem = MeshFileNames.Stem(name);
        List<string> notes = MeshExporters.CommonNotes(document, carriesMotion: false);
        List<MeshExportFile> files = [];
        Dictionary<MeshImage, string> imageNames = new(ReferenceEqualityComparer.Instance);
        string[] names = MaterialNames(document);
        string mtl = WriteMtl(document, stem, names, files, imageNames, notes);
        if (document.Primitives.Any(p => p.Geometry.VertexColors is not null))
            notes.Add("Vertex colours were dropped; OBJ has no standard place for them.");

        StringBuilder obj = new();
        obj.Append("# HartsyInference\n");
        if (document.Materials.Count > 0)
            obj.Append("mtllib ").Append(stem).Append(".mtl\n");
        int vertexBase = 0, uvBase = 0, normalBase = 0, group = 0;
        foreach (MeshPrimitive primitive in document.Primitives)
        {
            Mesh g = primitive.Geometry;
            bool uvs = g.Uvs is not null, normals = g.Normals is not null;
            obj.Append("g part_").Append(group++).Append('\n');
            if ((uint)primitive.Material < (uint)document.Materials.Count)
                obj.Append("usemtl ").Append(names[primitive.Material]).Append('\n');
            for (int i = 0; i < g.Vertices.Length; i += 3)
                obj.Append("v ").Append(F(g.Vertices[i])).Append(' ').Append(F(g.Vertices[i + 1])).Append(' ').Append(F(g.Vertices[i + 2])).Append('\n');
            if (uvs)
            {
                for (int i = 0; i < g.Uvs!.Length; i += 2)
                    obj.Append("vt ").Append(F(g.Uvs[i])).Append(' ').Append(F(1f - g.Uvs[i + 1])).Append('\n');
            }
            if (normals)
            {
                for (int i = 0; i < g.Normals!.Length; i += 3)
                    obj.Append("vn ").Append(F(g.Normals[i])).Append(' ').Append(F(g.Normals[i + 1])).Append(' ').Append(F(g.Normals[i + 2])).Append('\n');
            }
            for (int t = 0; t < g.Indices.Length; t += 3)
            {
                obj.Append('f');
                for (int k = 0; k < 3; k++)
                {
                    int v = g.Indices[t + k];
                    obj.Append(' ').Append(vertexBase + v + 1);
                    if (uvs || normals)
                    {
                        obj.Append('/');
                        if (uvs)
                            obj.Append(uvBase + v + 1);
                        if (normals)
                            obj.Append('/').Append(normalBase + v + 1);
                    }
                }
                obj.Append('\n');
            }
            vertexBase += g.VertexCount;
            if (uvs)
                uvBase += g.VertexCount;
            if (normals)
                normalBase += g.VertexCount;
        }

        files.Insert(0, new MeshExportFile(stem + ".obj", "model/obj", Encoding.UTF8.GetBytes(obj.ToString())));
        if (document.Materials.Count > 0)
            files.Insert(1, new MeshExportFile(stem + ".mtl", "text/plain", Encoding.UTF8.GetBytes(mtl)));
        return new MeshExport(files, notes);
    }

    private static string WriteMtl(MeshDocument document, string stem, string[] names, List<MeshExportFile> files, Dictionary<MeshImage, string> imageNames, List<string> notes)
    {
        StringBuilder mtl = new();
        mtl.Append("# HartsyInference\n");
        bool droppedMetalRough = false;
        for (int m = 0; m < document.Materials.Count; m++)
        {
            MeshMaterial material = document.Materials[m];
            float[] c = material.BaseColorFactor;
            mtl.Append("newmtl ").Append(names[m]).Append('\n');
            mtl.Append("Kd ").Append(F(c[0])).Append(' ').Append(F(c[1])).Append(' ').Append(F(c[2])).Append('\n');
            // Only a blended material is see-through; a masked one is cut out by its alpha, which `d` cannot say.
            mtl.Append("d ").Append(F(material.AlphaMode == MeshAlphaMode.Blend ? c[3] : 1f)).Append('\n');
            mtl.Append("Pm ").Append(F(material.Metallic)).Append('\n');
            mtl.Append("Pr ").Append(F(material.Roughness)).Append('\n');
            float[] e = material.EmissiveFactor;
            if (e.Any(v => v > 0))
                mtl.Append("Ke ").Append(F(e[0])).Append(' ').Append(F(e[1])).Append(' ').Append(F(e[2])).Append('\n');
            Slot(mtl, "map_Kd", material.BaseColor, $"{stem}_{m}_basecolor", files, imageNames);
            // The PBR MTL extension names a normal map `norm`; most importers (Blender among them) only read `map_Bump`, so both are written.
            Slot(mtl, "norm", material.Normal, $"{stem}_{m}_normal", files, imageNames);
            Slot(mtl, "map_Bump", material.Normal, $"{stem}_{m}_normal", files, imageNames);
            Slot(mtl, "map_Ke", material.Emissive, $"{stem}_{m}_emissive", files, imageNames);
            droppedMetalRough |= material.MetallicRoughness is not null || material.Occlusion is not null;
            mtl.Append('\n');
        }
        if (droppedMetalRough)
            notes.Add("Metallic-roughness and occlusion textures were left out; MTL has no slot for the packed maps (the metallic and roughness factors are kept as Pm and Pr).");
        return mtl.ToString();
    }

    private static void Slot(StringBuilder mtl, string keyword, MeshImage? image, string name, List<MeshExportFile> files, Dictionary<MeshImage, string> imageNames)
    {
        if (image is null)
            return;
        if (!imageNames.TryGetValue(image, out string? file))
        {
            file = $"{name}.{image.Extension}";
            imageNames[image] = file;
            files.Add(new MeshExportFile(file, image.MimeType, image.Data));
        }
        mtl.Append(keyword).Append(' ').Append(file).Append('\n');
    }

    /// <summary>The material names an MTL file uses: the document's own, made safe for a one-word MTL entry and made unique, or <c>material_i</c> when a material has none.</summary>
    private static string[] MaterialNames(MeshDocument document)
    {
        string[] names = new string[document.Materials.Count];
        HashSet<string> used = [];
        for (int i = 0; i < names.Length; i++)
        {
            string clean = new([.. document.Materials[i].Name.Select(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_' or '.' ? c : '_')]);
            string name = clean.Length == 0 ? $"material_{i}" : clean;
            string candidate = name;
            for (int suffix = i; !used.Add(candidate); suffix++)
                candidate = $"{name}_{suffix}";
            names[i] = candidate;
        }
        return names;
    }

    private static string F(float value) => value.ToString("R", CultureInfo.InvariantCulture);
}
