using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Pipelines.LongCat;

/// <summary>One denoise call's per-step dump into the directory of <c>diagnostics.longcatStepDump</c>, with the file names and meaning of the pinned
/// reference's dump, as float32 <c>.npy</c>: <c>call{n}_step{i}_t</c> (the timestep), <c>_noise</c> (the model output with the sign the scheduler
/// takes) and <c>_latent</c> (the latents after the step, conditioning frames included). The engine computes the noisy frames only, so
/// <c>_noise</c> is <c>[1, C, noisy, H, W]</c> where a reference file may hold every frame. Writing reads both tensors back to the host.</summary>
internal sealed class LongCatStepDump
{
    private readonly string _directory;
    private readonly int _call;

    /// <summary>The dump of call number <paramref name="call"/> into <paramref name="directory"/>, which is created.</summary>
    public LongCatStepDump(string directory, int call)
    {
        _directory = directory;
        _call = call;
        Directory.CreateDirectory(directory);
    }

    /// <summary>Writes step <paramref name="step"/>. <paramref name="velocity"/> and <paramref name="state"/> are the loop's
    /// <c>[channels·noisy, height·width]</c> tensors, the state already stepped.</summary>
    public void Write(int step, float timestep, Tensor velocity, Tensor state, float[]? condLatents, int channels, int frames, int cond, int height,
        int width)
    {
        // The loop steps state += dt · (−velocity); the reference's noise_pred is that negated output.
        float[] noise = velocity.AsReadOnlySpan<float>().ToArray();
        for (int i = 0; i < noise.Length; i++)
            noise[i] = -noise[i];
        float[] latents = LongCatFrameOps.JoinFrames(condLatents, state.AsReadOnlySpan<float>().ToArray(), channels, frames, height * width, cond);
        Save(step, "t", [timestep], []);
        Save(step, "noise", noise, [1, channels, frames - cond, height, width]);
        Save(step, "latent", latents, [1, channels, frames, height, width]);
    }

    private void Save(int step, string name, ReadOnlySpan<float> values, ReadOnlySpan<int> shape) =>
        File.WriteAllBytes(Path.Combine(_directory, $"call{_call:D2}_step{step:D2}_{name}.npy"), NpyFile.Float32(values, shape));
}
