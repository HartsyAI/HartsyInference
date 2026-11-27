namespace HartsyInference.World.Camera;

/// <summary>Plücker ray-map construction for camera-conditioned world models: per pixel, a 6-channel <c>(H, W, 6)</c>
/// map of the camera ray. The channel order is an explicit <see cref="PluckerLayout"/> and the pixel-center offset is an
/// explicit argument, because the upstream models differ on both (the existing callers sample at 0.5, SANA-WM's
/// <c>compute_raymap</c> at 0).</summary>
public static class PluckerEmbedding
{
    /// <summary>Channels per pixel.</summary>
    public const int Channels = 6;

    /// <summary>Pixel-center offset used by the <c>moment</c> overload.</summary>
    public const float PixelCenterOffset = 0.5f;

    /// <summary>Fills <paramref name="destination"/> (<c>height·width·6</c> floats, (h, w, c) layout) with the ray under
    /// <paramref name="cameraToWorld"/> (row-major 4×4) sampled at pixel centers <c>(x + 0.5, y + 0.5)</c>. The first triple is
    /// the origin (<paramref name="moment"/>=false) or the moment <c>o × d</c> (<paramref name="moment"/>=true); the second is
    /// the unit world-space direction.</summary>
    public static void Compute(ReadOnlySpan<float> cameraToWorld,
        float fx, float fy, float cx, float cy, int height, int width, Span<float> destination, bool moment = false)
    {
        Compute(cameraToWorld, fx, fy, cx, cy, height, width, destination, PixelCenterOffset,
            moment ? PluckerLayout.MomentDirection : PluckerLayout.OriginDirection);
    }

    /// <summary>Fills <paramref name="destination"/> (<c>height·width·6</c> floats, (h, w, c) layout) with the ray under
    /// <paramref name="cameraToWorld"/> (row-major 4×4) and the pinhole intrinsics. Pixel <c>(x, y)</c> is sampled at
    /// <c>(x + pixelOffset, y + pixelOffset)</c>; channels are ordered by <paramref name="layout"/>.</summary>
    public static void Compute(ReadOnlySpan<float> cameraToWorld,
        float fx, float fy, float cx, float cy, int height, int width, Span<float> destination,
        float pixelOffset, PluckerLayout layout)
    {
        if (cameraToWorld.Length < 16) throw new ArgumentException("cameraToWorld must hold 16 floats.", nameof(cameraToWorld));
        if (height < 0 || width < 0) throw new ArgumentOutOfRangeException(nameof(height), "Raymap size must be non-negative.");
        if (destination.Length < height * width * Channels)
            throw new ArgumentException($"destination must hold {height * width * Channels} floats.", nameof(destination));
        if (fx == 0 || fy == 0) throw new ArgumentException("focal lengths must be non-zero.");
        if (!float.IsFinite(pixelOffset)) throw new ArgumentOutOfRangeException(nameof(pixelOffset), pixelOffset, "Pixel offset must be finite.");
        if ((uint)layout > (uint)PluckerLayout.DirectionMoment)
            throw new ArgumentOutOfRangeException(nameof(layout), layout, "Unknown Plücker layout.");

        CameraRayFrame frame = CameraRayFrame.FromCameraToWorld(cameraToWorld);
        for (int yPix = 0; yPix < height; yPix++)
        {
            float ys = (yPix + pixelOffset - cy) / fy;
            for (int xPix = 0; xPix < width; xPix++)
            {
                float xs = (xPix + pixelOffset - cx) / fx;
                float inv = 1f / MathF.Sqrt(xs * xs + ys * ys + 1f);
                float dxc = xs * inv, dyc = ys * inv, dzc = inv;
                (float dwx, float dwy, float dwz) = frame.RotateToWorld(dxc, dyc, dzc);
                WritePixel(destination, (yPix * width + xPix) * Channels, layout,
                    frame.OriginX, frame.OriginY, frame.OriginZ, dwx, dwy, dwz);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void WritePixel(Span<float> destination, int baseIdx, PluckerLayout layout,
        float ox, float oy, float oz, float dx, float dy, float dz)
    {
        // Plücker moment m = o × d.
        float mx = oy * dz - oz * dy;
        float my = oz * dx - ox * dz;
        float mz = ox * dy - oy * dx;
        if (layout == PluckerLayout.DirectionMoment)
        {
            destination[baseIdx + 0] = dx; destination[baseIdx + 1] = dy; destination[baseIdx + 2] = dz;
            destination[baseIdx + 3] = mx; destination[baseIdx + 4] = my; destination[baseIdx + 5] = mz;
            return;
        }
        float px = ox, py = oy, pz = oz;
        if (layout == PluckerLayout.MomentDirection) { px = mx; py = my; pz = mz; }
        destination[baseIdx + 0] = px; destination[baseIdx + 1] = py; destination[baseIdx + 2] = pz;
        destination[baseIdx + 3] = dx; destination[baseIdx + 4] = dy; destination[baseIdx + 5] = dz;
    }
}
