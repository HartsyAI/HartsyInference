using HartsyInference.Core.Backends;

namespace HartsyInference.Engine.Recipes;

/// <summary>A video recipe that knows when the engine's device cannot run a checkpoint at all. Planning asks it for an otherwise valid plan whose
/// checkpoint is on disk, which constructs the backend.</summary>
internal interface IVideoDeviceCheck
{
    /// <summary>Why <paramref name="backend"/> cannot run the checkpoint at <paramref name="checkpointPath"/>, as the user reads it, or null.
    /// Reads no tensor data.</summary>
    string? DeviceRefusal(IBackend backend, string checkpointPath);
}
