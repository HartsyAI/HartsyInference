namespace HartsyInference.World.Camera;

/// <summary>Plücker ray-map construction for camera-conditioned world models: per pixel, a 6-channel <c>(H, W, 6)</c>
/// map of the camera ray. Two layouts are supported via <paramref name="moment"/>:
/// <list type="bullet">
/// <item><b>Default (<c>moment=false</c>)</b> — <c>[o_x, o_y, o_z, d_x, d_y, d_z]</c> (origin + unit direction).
/// Matches Matrix-Game-3 <c>utils/cam_utils.py:get_plucker_embeddings</c> (<c>cat([rays_o, rays_d])</c>).</item>
/// <item><b><c>moment=true</c></b> — <c>[m_x, m_y, m_z, d_x, d_y, d_z]</c> where <c>m = o × d</c> (the true Plücker
/// moment). Matches Hunyuan-GameCraft <c>ray_condition</c> (<c>cat([cross(rays_o, rays_d), rays_d])</c>).</item>
/// </list>
/// The two upstream models genuinely differ here, so the layout is an explicit per-call choice.</summary>
public static class PluckerEmbedding
{
    /// <summary>Channels per pixel.</summary>
    public const int Channels = 6;

    /// <summary>Fills <paramref name="destination"/> (<c>height·width·6</c> floats, (h, w, c) layout) with the per-pixel
    /// ray under <paramref name="cameraToWorld"/> (row-major 4×4) and the pinhole intrinsics. Pixel centers are sampled
    /// at <c>(x + 0.5, y + 0.5)</c>. The first 3 channels are the ray origin (<paramref name="moment"/>=false) or the
    /// Plücker moment <c>o × d</c> (<paramref name="moment"/>=true); the last 3 are the unit world-space direction.</summary>
    public static void Compute(ReadOnlySpan<float> cameraToWorld,
        float fx, float fy, float cx, float cy, int height, int width, Span<float> destination, bool moment = false)
    {
        if (cameraToWorld.Length < 16) throw new ArgumentException("cameraToWorld must hold 16 floats.", nameof(cameraToWorld));
        if (destination.Length < height * width * Channels)
            throw new ArgumentException($"destination must hold {height * width * Channels} floats.", nameof(destination));
        if (fx == 0 || fy == 0) throw new ArgumentException("focal lengths must be non-zero.");

        float ox = cameraToWorld[3], oy = cameraToWorld[7], oz = cameraToWorld[11];
        for (int yPix = 0; yPix < height; yPix++)
        {
            float ys = (yPix + 0.5f - cy) / fy;
            for (int xPix = 0; xPix < width; xPix++)
            {
                float xs = (xPix + 0.5f - cx) / fx;
                float inv = 1f / MathF.Sqrt(xs * xs + ys * ys + 1f);
                float dxc = xs * inv, dyc = ys * inv, dzc = inv;
                // World direction = R_c2w · d_cam.
                float dwx = cameraToWorld[0] * dxc + cameraToWorld[1] * dyc + cameraToWorld[2] * dzc;
                float dwy = cameraToWorld[4] * dxc + cameraToWorld[5] * dyc + cameraToWorld[6] * dzc;
                float dwz = cameraToWorld[8] * dxc + cameraToWorld[9] * dyc + cameraToWorld[10] * dzc;
                int baseIdx = (yPix * width + xPix) * Channels;
                if (moment)
                {
                    // Plücker moment m = o × d (GameCraft ray_condition layout).
                    destination[baseIdx + 0] = oy * dwz - oz * dwy;
                    destination[baseIdx + 1] = oz * dwx - ox * dwz;
                    destination[baseIdx + 2] = ox * dwy - oy * dwx;
                }
                else
                {
                    destination[baseIdx + 0] = ox;
                    destination[baseIdx + 1] = oy;
                    destination[baseIdx + 2] = oz;
                }
                destination[baseIdx + 3] = dwx;
                destination[baseIdx + 4] = dwy;
                destination[baseIdx + 5] = dwz;
            }
        }
    }
}
