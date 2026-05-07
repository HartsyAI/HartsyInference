namespace SharpInference.ModelHandler.Gguf.Codecs;

/// <summary>Helpers for K-quant quantization. Implements simplified ggml `make_qkx2_quants` (initial pass only — no iterative refinement search). Produces valid GGUF blocks with ~5% PPL gap vs canonical ggml output. Sufficient for authoring our own quants; users wanting llama.cpp-bit-identical output should use llama.cpp's `quantize` tool instead.</summary>
internal static unsafe class QkxQuantizer
{
    /// <summary>Finds <c>(scale, -min)</c> for an n-element block such that <c>x[i] ≈ scale * q[i] + (-(-min)) = scale * q[i] + min_offset</c> with <c>q ∈ [0..nmax]</c>. Returns <c>scale</c> and writes <c>negativeMin</c> = -min (so reconstruction is <c>x = scale * q - negativeMin</c>, matching ggml's layout). Also writes the <c>L</c> 0..nmax quants.</summary>
    public static float MakeQkx2Quants(int n, int nmax, float* x, byte* L, out float negativeMin)
    {
        float min = x[0];
        float max = x[0];
        for (int i = 1; i < n; i++)
        {
            if (x[i] < min) min = x[i];
            if (x[i] > max) max = x[i];
        }
        if (min > 0) min = 0;
        if (max == min)
        {
            for (int i = 0; i < n; i++) L[i] = 0;
            negativeMin = -min;
            return 0f;
        }

        float iscale = nmax / (max - min);
        float scale = 1f / iscale;
        for (int i = 0; i < n; i++)
        {
            int l = NearestInt(iscale * (x[i] - min));
            L[i] = (byte)Math.Clamp(l, 0, nmax);
        }
        negativeMin = -min;
        return scale;
    }

    /// <summary>Finds the best symmetric scale for an n-element block such that <c>x[i] ≈ scale * q[i]</c> with <c>q ∈ [-nmax..nmax]</c>. Used by Q6_K and the legacy Q4_0/Q5_0/Q8_0 paths. Returns the scale.</summary>
    public static float MakeSymmetricScale(int n, int nmax, float* x, sbyte* L)
    {
        float maxAbs = 0f;
        for (int i = 0; i < n; i++)
        {
            float a = MathF.Abs(x[i]);
            if (a > maxAbs) maxAbs = a;
        }
        if (maxAbs == 0f)
        {
            for (int i = 0; i < n; i++) L[i] = 0;
            return 0f;
        }
        float scale = maxAbs / nmax;
        float invScale = 1f / scale;
        for (int i = 0; i < n; i++)
        {
            int l = NearestInt(x[i] * invScale);
            L[i] = (sbyte)Math.Clamp(l, -nmax, nmax);
        }
        return scale;
    }

    /// <summary>Packs 16 6-bit (scale, min) pairs into 12 bytes per the canonical ggml layout — inverse of <see cref="GgufCodecHelpers.GetScaleMinK4"/>. Sub-blocks 0..3 use bytes [j] (scale low6) and [j+4] (min low6); sub-blocks 4..7 pack low4 + min-low4 into byte [j+4] and stash the high2 bits into bytes [j-4] and [j].</summary>
    public static void PackScaleMinK4(byte sc, byte mm, int j, byte* q)
    {
        if (j < 4)
        {
            q[j] = (byte)((q[j] & 0xC0) | (sc & 0x3F));
            q[j + 4] = (byte)((q[j + 4] & 0xC0) | (mm & 0x3F));
        }
        else
        {
            q[j + 4] = (byte)((sc & 0x0F) | ((mm & 0x0F) << 4));
            q[j - 4] = (byte)((q[j - 4] & 0x3F) | (((sc >> 4) & 0x03) << 6));
            q[j - 0] = (byte)((q[j - 0] & 0x3F) | (((mm >> 4) & 0x03) << 6));
        }
    }

    public static int NearestInt(float v) => (int)MathF.Round(v);
}
