using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.StyleTts2;

/// <summary>StyleTTS2 <c>SourceModuleHnNSF</c>/<c>SineGen</c> harmonic source (deterministic, no additive NSF
/// noise): 9 harmonics of the frame-level F0, with StyleTTS2's specific phase path — F0 is nearest-upsampled to
/// audio rate, phase is <c>cumsum</c>'d at FRAME rate, then <c>(phase · upsample_scale)</c> is <b>linearly</b>
/// interpolated up (align_corners=False), <c>sin</c>'d × 0.1, gated by voiced/unvoiced, then merged by the
/// trained <c>l_linear</c> (9→1) + tanh. Matches <c>Modules/hifigan.py</c> with the random initial-phase set to
/// zero (that term is seed-random and perceptually inert; zeroing it makes the source deterministic).</summary>
public static unsafe class StyleSineGen
{
    private const int Harmonics = 9;
    private const float SineAmp = 0.1f;
    private const float VoicedThreshold = 10f;

    /// <summary>Returns the merged 1-channel harmonic source at audio rate (<c>frames·upScale</c> samples).</summary>
    public static float[] Source(Tensor f0, int upScale, int sampleRate, Tensor mergeW, Tensor mergeB)
    {
        int frames = (int)f0.Shape[f0.Shape.Rank - 1];
        int tAudio = frames * upScale;
        float* fp = (float*)f0.DataPointer;
        float* mW = (float*)mergeW.DataPointer;
        float mB = ((float*)mergeB.DataPointer)[0];

        // Frame-rate phase: rad_down[j,h] = (f0[j]·(h+1)/sr) mod 1; phase[j,h] = 2π·Σ_{k≤j} rad, then ×upScale.
        double[] phaseScaled = new double[(long)frames * Harmonics];   // (phase · upScale), frame rate
        double[] cum = new double[Harmonics];
        for (int j = 0; j < frames; j++)
        {
            double hz = fp[j];
            for (int h = 0; h < Harmonics; h++)
            {
                double rad = (hz * (h + 1) / sampleRate) % 1.0;
                cum[h] += rad;                                     // inclusive cumsum
                phaseScaled[(long)j * Harmonics + h] = cum[h] * 2.0 * Math.PI * upScale;
            }
        }

        float[] merged = new float[tAudio];
        for (int t = 0; t < tAudio; t++)
        {
            // Linear upsample (align_corners=False): source coord in frame index space.
            double src = (t + 0.5) / upScale - 0.5;
            int j0 = (int)Math.Floor(src);
            double frac = src - j0;
            int j0c = Math.Clamp(j0, 0, frames - 1);
            int j1c = Math.Clamp(j0 + 1, 0, frames - 1);
            // uv from the nearest-upsampled F0 (piecewise-constant per frame block).
            int frameOfT = Math.Clamp((int)Math.Floor((t + 0.5) / upScale), 0, frames - 1);
            float uv = fp[frameOfT] > VoicedThreshold ? 1f : 0f;

            double lin = mB;
            for (int h = 0; h < Harmonics; h++)
            {
                double p0 = phaseScaled[(long)j0c * Harmonics + h];
                double p1 = phaseScaled[(long)j1c * Harmonics + h];
                double phase = p0 + (p1 - p0) * frac;
                float sine = (float)Math.Sin(phase) * SineAmp;
                lin += mW[h] * (sine * uv);
            }
            merged[t] = MathF.Tanh((float)lin);
        }
        return merged;
    }
}
