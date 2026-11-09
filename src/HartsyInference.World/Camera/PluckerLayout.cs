namespace HartsyInference.World.Camera;

/// <summary>Channel order of a 6-channel Plücker ray map: the first triple, then the second.</summary>
public enum PluckerLayout
{
    /// <summary>Origin <c>o</c>, then unit direction <c>d</c>. Matrix-Game 3.0 (<c>cat([rays_o, rays_d])</c>).</summary>
    OriginDirection,

    /// <summary>Moment <c>m = o × d</c>, then unit direction <c>d</c>. Hunyuan-GameCraft (<c>ray_condition</c>).</summary>
    MomentDirection,

    /// <summary>Unit direction <c>d</c>, then moment <c>m = o × d</c>. SANA-WM (<c>compute_raymap</c>).</summary>
    DirectionMoment,
}
