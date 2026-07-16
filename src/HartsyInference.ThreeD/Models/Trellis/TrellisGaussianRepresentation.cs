using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Models.Trellis;

/// <summary>Converts the SLAT Gaussian decoder's 448-dim per-voxel output into a <see cref="GaussianSplatCloud"/>
/// (TRELLIS <c>to_representation</c>). Per active voxel, 32 gaussians: position = voxel-center + tanh(offset +
/// hammersley perturbation)·½·voxel_size/res; scaling/rotation/opacity/features_dc reshaped per gaussian. Fields are
/// stored raw (log-scale, logit-opacity, quaternion) — <see cref="Io.PlyWriter.SaveSplats"/> writes the 3DGS PLY.
/// A y-up→z-up transform ([x,y,z]→[x,−z,y], quaternion rotated 90° about X) matches the reference viewing.</summary>
public static unsafe class TrellisGaussianRepresentation
{
    private const int NumGaussians = 32, Resolution = 64;
    private const float VoxelSize = 1.5f, MinKernel = 0.0009f;

    public static GaussianSplatCloud Build(SparseTensor netOut)
    {
        int nv = netOut.Count;
        int splats = nv * NumGaussians;
        float scaleBias = SoftplusInverse(0.004f);          // ≈ −5.52
        float opacityBias = MathF.Log(0.1f / 0.9f);          // logit(0.1) ≈ −2.197
        float offScale = 0.5f * VoxelSize / Resolution;      // tanh(offset)·this

        // hammersley perturbation [32,3] = atanh((hammersley·2−1)/voxel_size).
        float[] pert = new float[NumGaussians * 3];
        for (int g = 0; g < NumGaussians; g++)
        {
            float[] h = { (float)g / NumGaussians, RadicalInverse(2, g), RadicalInverse(3, g) };
            for (int c = 0; c < 3; c++) pert[g * 3 + c] = Atanh((h[c] * 2f - 1f) / VoxelSize);
        }

        float[] pos = new float[splats * 3], scl = new float[splats * 3], rot = new float[splats * 4], op = new float[splats], sh = new float[splats * 3];
        float* f = (float*)netOut.Feats.DataPointer;
        // layout ranges: _xyz [0,96) _features_dc [96,192) _scaling [192,288) _rotation [288,416) _opacity [416,448)
        const int rXyz = 0, rDc = 96, rScl = 192, rRot = 288, rOp = 416;
        // y-up→z-up quaternion (90° about X): q_R = [cos45, sin45, 0, 0].
        const float c45 = 0.70710678f;
        for (int v = 0; v < nv; v++)
        {
            float* row = f + (long)v * 448;
            float cx = (netOut.Coords[v * 4 + 1] + 0.5f) / Resolution - 0.5f;   // voxel centre, aabb-shifted (get_xyz)
            float cy = (netOut.Coords[v * 4 + 2] + 0.5f) / Resolution - 0.5f;
            float cz = (netOut.Coords[v * 4 + 3] + 0.5f) / Resolution - 0.5f;
            for (int g = 0; g < NumGaussians; g++)
            {
                long s = (long)v * NumGaussians + g;
                // position: centre + tanh(offset + perturb)·offScale, then y-up transform [x,-z,y].
                float px = cx + MathF.Tanh(row[rXyz + g * 3 + 0] + pert[g * 3 + 0]) * offScale;
                float py = cy + MathF.Tanh(row[rXyz + g * 3 + 1] + pert[g * 3 + 1]) * offScale;
                float pz = cz + MathF.Tanh(row[rXyz + g * 3 + 2] + pert[g * 3 + 2]) * offScale;
                pos[s * 3 + 0] = px; pos[s * 3 + 1] = -pz; pos[s * 3 + 2] = py;
                // features_dc (SH DC, 3), stored raw.
                for (int c = 0; c < 3; c++) sh[s * 3 + c] = row[rDc + g * 3 + c];
                // scaling: log(sqrt(softplus(_scaling+bias)² + min²)).
                for (int c = 0; c < 3; c++) { float sp = Softplus(row[rScl + g * 3 + c] + scaleBias); scl[s * 3 + c] = 0.5f * MathF.Log(sp * sp + MinKernel * MinKernel); }
                // opacity: logit = _opacity + bias.
                op[s] = row[rOp + g] + opacityBias;
                // rotation: (_rotation + [1,0,0,0]) rotated by q_R (y-up transform).
                float qw = row[rRot + g * 4 + 0] + 1f, qx = row[rRot + g * 4 + 1], qy = row[rRot + g * 4 + 2], qz = row[rRot + g * 4 + 3];
                rot[s * 4 + 0] = c45 * qw - c45 * qx; rot[s * 4 + 1] = c45 * qx + c45 * qw;
                rot[s * 4 + 2] = c45 * qy + c45 * qz; rot[s * 4 + 3] = c45 * qz - c45 * qy;   // q_R ⊗ q
            }
        }

        return new GaussianSplatCloud
        {
            Positions = pos, Scales = scl, Rotations = rot, Opacities = op,
            ShCoefficients = sh, ShCoeffsPerSplat = 1
        };
    }


    private static float Softplus(float x) => x > 20f ? x : MathF.Log(1f + MathF.Exp(x));
    private static float SoftplusInverse(float x) => x + MathF.Log(1f - MathF.Exp(-x));   // = x + log(−expm1(−x))
    private static float Atanh(float x) => 0.5f * MathF.Log((1f + x) / (1f - x));
    private static float RadicalInverse(int baseN, int n)
    {
        float val = 0f, invBase = 1f / baseN, invBaseN = invBase;
        while (n > 0) { int digit = n % baseN; val += digit * invBaseN; n /= baseN; invBaseN *= invBase; }
        return val;
    }
}
