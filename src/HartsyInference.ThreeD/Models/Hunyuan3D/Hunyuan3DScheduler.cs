namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>The sigma schedule a Hunyuan3D-2 checkpoint was trained to sample with.</summary>
public enum Hunyuan3DScheduler
{
    /// <summary>Evenly spaced sigmas from 0 to 1 (<c>FlowMatchEulerDiscreteScheduler</c>, shift 1).</summary>
    FlowMatchEuler,

    /// <summary>Sigmas picked from a <c>pcm_timesteps</c> table (<c>ConsistencyFlowMatchEulerDiscreteScheduler</c>, the Turbo builds).</summary>
    ConsistencyFlowMatchEuler,
}
