using HartsyInference.Core.Backends;
using HartsyInference.Core.Runtime;

namespace HartsyInference.Core.MemoryManagement;

/// <summary>Resolves a <see cref="LeverState"/> to the concrete yes/no a call site needs, falling back to the legacy environment variable when the lever is <see cref="LeverState.Auto"/>.</summary>
/// <remarks>The fallback is what keeps existing deployments byte-identical: a policy that never pins a lever behaves
/// exactly as the environment variable did, while a tier or a request override takes precedence over it. Resolved per
/// call rather than cached in a <c>static readonly</c> — a host may set its policy after a warm-up generation has
/// already run, and a cached first answer would make the setting appear to do nothing. Every call site here is
/// once-per-generation or once-per-phase, never per step.</remarks>
public static class VramLevers
{
    /// <summary>Legacy switch for keeping a pipeline's weights on the device between generations.</summary>
    public const string KeepModelsVariable = "HARTSY_KEEP_MODELS";

    /// <summary>Whether <paramref name="backend"/>'s pipelines keep their weights resident between generations.</summary>
    public static bool KeepResident(IBackend? backend) => KeepResident(VramPolicyRegistry.Resolve(backend));

    /// <summary>Whether <paramref name="policy"/> keeps weights resident between generations; for callers that already resolved it.</summary>
    public static bool KeepResident(VramPolicy policy)
    {
        ArgumentNullException.ThrowIfNull(policy);
        return Resolve(policy.KeepResident, KeepModelsVariable, defaultOn: true);
    }

    /// <summary>The lever's concrete value: an explicit <see cref="LeverState.On"/>/<see cref="LeverState.Off"/> wins, and <see cref="LeverState.Auto"/> defers to <paramref name="environmentVariable"/>.</summary>
    public static bool Resolve(LeverState state, string environmentVariable, bool defaultOn) => state switch
    {
        LeverState.On => true,
        LeverState.Off => false,
        _ => EnvSwitch.IsEnabled(environmentVariable, defaultOn),
    };
}
