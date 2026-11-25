namespace HartsyInference.ModelAssets.Duo;

/// <summary>Where one diffusers-named LoRA module lands in the backbone: the weight key, which of <paramref name="Slots"/> equal row blocks of it the module owns, and whether its B rows are in the <c>[value; gate]</c> order that the backbone's <c>[gate; value]</c> <c>mlp.fc1</c> swaps.</summary>
public readonly record struct DuoLoraTarget(string BackboneKey, int Slot, int Slots, bool SwapHalves);
