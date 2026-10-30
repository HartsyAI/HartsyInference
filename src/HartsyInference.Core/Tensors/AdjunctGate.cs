namespace HartsyInference.Core.Tensors;

/// <summary>An on/off switch and a strength multiplier shared by the <see cref="LowRankAdjunctTerm"/>s of one LoRA, so a
/// LoRA can be live for some calls and absent from others on the same resident weights.</summary>
/// <remarks><para>A gated term contributes to a GEMM only while its gate is enabled. Disabled, the term is skipped
/// entirely, so the output is the base projection's, bit for bit. The factors stay attached and resident either way.</para>
/// <para>Single owner: the code that enables the gate must serialize the generations that read it. Reads are plain
/// fields, because the GEMM path reads the gate once per projection per step.</para>
/// <para>A host that captures a graph bakes the accumulate-or-skip choice and the term scales into the graph, so its
/// graph signature has to include the gate state (Anima captures none).</para></remarks>
public sealed class AdjunctGate
{
    private bool _enabled;
    private float _scale = 1.0f;

    /// <summary>Whether the terms sharing this gate are added to their GEMM results right now.</summary>
    public bool Enabled => _enabled;

    /// <summary>Multiplier applied to every sharing term's scale while the gate is enabled.</summary>
    public float Scale => _scale;

    /// <summary>Enables the gate at <paramref name="scale"/> until the returned scope is disposed. The previous state is
    /// restored on dispose, so scopes nest.</summary>
    public IDisposable Enable(float scale)
    {
        if (!float.IsFinite(scale) || scale < 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(scale), scale, "An adjunct gate scale must be finite and non-negative.");
        }
        Scope scope = new(this, _enabled, _scale);
        _enabled = true;
        _scale = scale;
        return scope;
    }

    private sealed class Scope(AdjunctGate gate, bool enabled, float scale) : IDisposable
    {
        private bool _disposed;

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            gate._enabled = enabled;
            gate._scale = scale;
        }
    }
}
