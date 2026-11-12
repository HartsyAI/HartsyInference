namespace HartsyInference.Engine.Planning;

/// <summary>Whether a mesh planning diagnostic is informational, cautionary, or prevents the run.</summary>
public enum MeshPlanIssueSeverity
{
    /// <summary>Information only.</summary>
    Info = 0,

    /// <summary>The request runs but deserves attention (a field that has no effect, a model that is not verified).</summary>
    Warning = 1,

    /// <summary>The request must not start: the run would be refused or fail.</summary>
    Error = 2,
}
