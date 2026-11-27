namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>
/// Gives a triangle soup consistent winding: across every edge that exactly two triangles share, the two traverse it in opposite
/// directions, and each connected patch has a positive signed volume so its triangles face outward. A dual-grid quad is wound the
/// same way whichever side of the surface is inside, so raw TRELLIS.2 meshes need this before their vertex normals are meaningful;
/// upstream does it in <c>unify_face_orientations</c>.
/// </summary>
public static class FaceOrientation
{
    /// <summary>Re-winds <paramref name="mesh"/> in place and returns how many triangles it flipped.</summary>
    public static int Unify(Mesh mesh)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        int[] indices = mesh.Indices;
        int faces = indices.Length / 3;
        if (faces == 0)
            return 0;

        int[] neighbor = LinkManifoldEdges(indices, faces, out bool[] opposed);
        bool[] flip = new bool[faces];
        int[] patch = new int[faces];
        int patches = PropagateWinding(neighbor, opposed, flip, patch, faces);
        FlipInwardPatches(mesh, flip, patch, patches);

        int flipped = 0;
        for (int f = 0; f < faces; f++)
        {
            if (!flip[f])
                continue;
            (indices[f * 3 + 1], indices[f * 3 + 2]) = (indices[f * 3 + 2], indices[f * 3 + 1]);
            flipped++;
        }

        return flipped;
    }

    // Slot f*3+k holds the triangle across edge k (vertex k to k+1), or -1 when the edge is open or shared by three or more.
    // opposed[slot] is true when both triangles traverse the edge the same way, so one of them must flip.
    private static int[] LinkManifoldEdges(int[] indices, int faces, out bool[] opposed)
    {
        int slots = faces * 3;
        long[] keys = new long[slots];
        int[] slotOf = new int[slots];
        bool[] forward = new bool[slots];
        for (int slot = 0; slot < slots; slot++)
        {
            int face = slot / 3, k = slot % 3;
            int a = indices[face * 3 + k], b = indices[face * 3 + (k + 1) % 3];
            forward[slot] = a < b;
            keys[slot] = a < b ? ((long)a << 32) | (uint)b : ((long)b << 32) | (uint)a;
            slotOf[slot] = slot;
        }

        Array.Sort(keys, slotOf);
        int[] neighbor = new int[slots];
        Array.Fill(neighbor, -1);
        opposed = new bool[slots];
        for (int i = 0; i < slots;)
        {
            int j = i + 1;
            while (j < slots && keys[j] == keys[i])
                j++;
            if (j - i == 2)
            {
                int s0 = slotOf[i], s1 = slotOf[i + 1];
                neighbor[s0] = s1 / 3;
                neighbor[s1] = s0 / 3;
                bool sameDirection = forward[s0] == forward[s1];
                opposed[s0] = sameDirection;
                opposed[s1] = sameDirection;
            }

            i = j;
        }

        return neighbor;
    }

    // Breadth-first over the linked triangles; the first triangle of each patch keeps its winding.
    private static int PropagateWinding(int[] neighbor, bool[] opposed, bool[] flip, int[] patch, int faces)
    {
        Array.Fill(patch, -1);
        int[] queue = new int[faces];
        int patches = 0;
        for (int seed = 0; seed < faces; seed++)
        {
            if (patch[seed] >= 0)
                continue;
            int head = 0, tail = 0;
            queue[tail++] = seed;
            patch[seed] = patches;
            while (head < tail)
            {
                int f = queue[head++];
                for (int k = 0; k < 3; k++)
                {
                    int g = neighbor[f * 3 + k];
                    if (g < 0 || patch[g] >= 0)
                        continue;
                    patch[g] = patches;
                    flip[g] = flip[f] ^ opposed[f * 3 + k];
                    queue[tail++] = g;
                }
            }

            patches++;
        }

        return patches;
    }

    // The signed volume of a patch is taken about its own centroid so an open patch does not depend on where the mesh sits.
    private static void FlipInwardPatches(Mesh mesh, bool[] flip, int[] patch, int patches)
    {
        float[] v = mesh.Vertices;
        int[] idx = mesh.Indices;
        int faces = idx.Length / 3;
        double[] cx = new double[patches], cy = new double[patches], cz = new double[patches];
        int[] count = new int[patches];
        for (int f = 0; f < faces; f++)
        {
            int p = patch[f];
            for (int k = 0; k < 3; k++)
            {
                int at = idx[f * 3 + k] * 3;
                cx[p] += v[at];
                cy[p] += v[at + 1];
                cz[p] += v[at + 2];
            }

            count[p] += 3;
        }

        double[] volume = new double[patches];
        for (int f = 0; f < faces; f++)
        {
            int p = patch[f];
            double ox = cx[p] / count[p], oy = cy[p] / count[p], oz = cz[p] / count[p];
            int ia = idx[f * 3] * 3, ib = idx[f * 3 + 1] * 3, ic = idx[f * 3 + 2] * 3;
            double ax = v[ia] - ox, ay = v[ia + 1] - oy, az = v[ia + 2] - oz;
            double bx = v[ib] - ox, by = v[ib + 1] - oy, bz = v[ib + 2] - oz;
            double ccx = v[ic] - ox, ccy = v[ic + 1] - oy, ccz = v[ic + 2] - oz;
            double det = ax * (by * ccz - bz * ccy) - ay * (bx * ccz - bz * ccx) + az * (bx * ccy - by * ccx);
            volume[p] += flip[f] ? -det : det;
        }

        for (int f = 0; f < faces; f++)
        {
            if (volume[patch[f]] < 0.0)
                flip[f] = !flip[f];
        }
    }
}
