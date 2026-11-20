using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora;

namespace HartsyInference.ThreeD.Models.Idol;

/// <summary>Idol-turbo's two phase experts carried on the denoiser's weights: a rank-32 slow LoRA for the high-noise steps and a fast one for the low-noise steps (the distillation of
/// guidance and of the 50-step schedule into 5 guidance-free steps). Each is a runtime adjunct on every block matrix (attention qkv and out, both MLP layers of the refiners and the 24 joint
/// blocks), switched by its own <see cref="AdjunctGate"/>, so the base F16 weights stay as they are and one set of weights serves both phases.</summary>
/// <remarks>The Comfy-format files are pre-scaled for strength 1.0 (idol-turbo's own file carries LoRA alpha 128 over rank 32, a factor of 4, which they already include).
/// Build this over loaded <see cref="IdolWeights"/>, make the <see cref="IdolDenoiser"/> over <see cref="Weights"/>, and open <see cref="Phase"/> around every forward pass.</remarks>
public sealed class IdolTurboAdapters : IDisposable
{
    /// <summary>The slow expert runs while the shifted time is at or above this, the fast one below it (idol-turbo's <c>PHASE_THRESHOLD</c>, in Idol's t axis where 1 is pure noise).</summary>
    public const float PhaseThreshold = 0.5f;

    /// <summary>Idol-turbo's default step count; it takes no guidance.</summary>
    public const int DefaultSteps = 5;

    private readonly LoraStack _stack = new();
    /// <summary>The scale of an embedded expert: LoRA alpha 128 over rank 32.</summary>
    public const float EmbeddedScale = 4f;

    private readonly LoraFile? _slowFile;
    private readonly LoraFile? _fastFile;
    private readonly AdjunctGate _slow = new();
    private readonly AdjunctGate _fast = new();
    private bool _disposed;

    private readonly List<Tensor> _aliases = [];

    // Embedded experts: one adjunct per block matrix with a slow and a fast term at scale alpha / rank = 128 / 32.
    private IdolTurboAdapters(IdolWeights weights)
    {
        _slowFile = null;
        _fastFile = null;
        Dictionary<string, Tensor> carried = new(weights.Tensors, StringComparer.Ordinal);
        foreach ((string module, IdolWeights.PhaseFactors f) in weights.Experts!)
        {
            LowRankAdjunct adjunct = new()
            {
                Terms =
                [
                    new LowRankAdjunctTerm { Down = f.SlowA, Up = f.SlowB, Scale = EmbeddedScale, Gate = _slow },
                    new LowRankAdjunctTerm { Down = f.FastA, Up = f.FastB, Scale = EmbeddedScale, Gate = _fast },
                ],
            };
            Tensor alias = weights[module + ".weight"].WithLowRankAdjunct(adjunct);
            _aliases.Add(alias);
            carried[module + ".weight"] = alias;
        }

        Weights = IdolWeights.FromTensors(weights.Config, carried);
    }

    private IdolTurboAdapters(LoraFile slow, LoraFile fast, IdolWeights weights, Core.Backends.IBackend backend)
    {
        _slowFile = slow;
        _fastFile = fast;
        try
        {
            Dictionary<string, Tensor> carried = new(weights.Tensors, StringComparer.Ordinal);
            _stack.AddGated(slow, _slow);
            _stack.AddGated(fast, _fast);
            // Both experts adapt the same modules: 112 distinct weights, 224 units.
            // Four modules (qkv, out, the two MLP layers) in each refiner block (noise and context) and each joint block.
            int modulesPerFile = (2 * weights.Config.RefinerBlocks + weights.Config.Blocks) * 4;
            int weightsReached = _stack.ApplyToWeights(backend, transformerWeights: carried);
            LoraStack.ApplyReport report = _stack.Report;
            if (weightsReached != modulesPerFile || report.Applied != 2 * modulesPerFile || report.SkippedMissing != 0 || report.SkippedShape != 0)
                throw new HartsyInferenceException(
                    $"The idol-turbo LoRAs reached {weightsReached} of {modulesPerFile} denoiser weights with {report.Applied} of {2 * modulesPerFile} units ({report.SkippedMissing} had no matching weight, {report.SkippedShape} the wrong shape); "
                    + "they are for the Comfy-Org/Idol denoiser (idol_fp16.safetensors) and its joint blocks and its noise and context refiners.");
            Weights = IdolWeights.FromTensors(weights.Config, carried);
        }
        catch
        {
            _stack.Dispose();
            throw;
        }
    }

    /// <summary>The base weights with both experts attached to their block matrices; borrowed tensors, so it needs no disposal of its own and must not outlive the weights it was built over.</summary>
    public IdolWeights Weights { get; private set; } = null!;

    /// <summary>Attaches the phase experts a checkpoint embeds in its block matrices (Budget3D-Idol-Turbo): the factors as they are, at LoRA alpha 128 over rank 32, a scale of 4.</summary>
    /// <exception cref="HartsyInferenceException">The checkpoint embeds no experts.</exception>
    public static IdolTurboAdapters AttachEmbedded(IdolWeights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (weights.Experts is null)
            throw new HartsyInferenceException("The checkpoint embeds no phase experts (slow_A, slow_B, fast_A, fast_B).");
        return new IdolTurboAdapters(weights);
    }

    /// <summary>Opens the files and attaches both experts to <paramref name="weights"/>' block matrices.</summary>
    /// <param name="slowPath">The Comfy-format slow expert (<c>idol_turbo_slow_comfy.safetensors</c>).</param>
    /// <param name="fastPath">The Comfy-format fast expert (<c>idol_turbo_fast_comfy.safetensors</c>).</param>
    /// <exception cref="HartsyInferenceException">A file does not match the denoiser: a module is missing or the wrong shape.</exception>
    public static IdolTurboAdapters Attach(Core.Backends.IBackend backend, IdolWeights weights, string slowPath, string fastPath)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(weights);
        LoraFile slow = LoraFile.Load(slowPath);
        LoraFile fast;
        try
        {
            fast = LoraFile.Load(fastPath);
        }
        catch
        {
            slow.Dispose();
            throw;
        }
        if (slow.Format != LoraFormat.Idol || fast.Format != LoraFormat.Idol)
        {
            string found = $"{Path.GetFileName(slowPath)} is {slow.Format}, {Path.GetFileName(fastPath)} is {fast.Format}";
            slow.Dispose();
            fast.Dispose();
            throw new HartsyInferenceException($"The idol-turbo LoRAs are not in the Idol format ({found}).");
        }
        try
        {
            return new IdolTurboAdapters(slow, fast, weights, backend);
        }
        catch
        {
            slow.Dispose();
            fast.Dispose();
            throw;
        }
    }

    /// <summary>Whether a pass at <paramref name="tModel"/> (Idol's <c>1000·t</c>) is in the slow phase.</summary>
    public static bool IsSlow(float tModel) => tModel / 1000f >= PhaseThreshold;

    /// <summary>The image stream of <paramref name="model"/> under each expert. The context refiner's four matrices per block carry the experts too, so the stream depends on the phase and is embedded once under each, not once for the picture as the base model does.</summary>
    public PhaseContexts EmbedContexts(IBackend backend, IdolDenoiser model, Tensor feature1, Tensor feature2)
    {
        ArgumentNullException.ThrowIfNull(model);
        Tensor slow, fast;
        using (Phase(1000f))
            slow = model.EmbedContext(backend, feature1, feature2);
        try
        {
            using (Phase(0f))
                fast = model.EmbedContext(backend, feature1, feature2);
        }
        catch
        {
            slow.Dispose();
            throw;
        }

        return new PhaseContexts(slow, fast);
    }

    /// <summary>The scope that runs one forward pass at <paramref name="tModel"/> (Idol's <c>1000·t</c>) with its phase's expert on: slow at or above <see cref="PhaseThreshold"/>, fast below.</summary>
    public IDisposable Phase(float tModel)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return (IsSlow(tModel) ? _slow : _fast).Enable(1f);
    }

    /// <summary>Releases the aliases the stack made and the two files.</summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _stack.Dispose();
        foreach (Tensor alias in _aliases)
            alias.Dispose();
        _slowFile?.Dispose();
        _fastFile?.Dispose();
    }
}
