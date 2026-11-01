namespace HartsyInference.Engine.Recipes;

/// <summary>A recipe that can attach a reference adapter at construction: a LoRA the reference path reads, switched by a gate per
/// generation. Requests that use a reference build a pipeline with the adapter attached; the adapter's name is part of the pipeline's
/// cache identity, so a request without one never reuses it.</summary>
public interface IReferenceAdapterRecipe
{
}
