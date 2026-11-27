# Krea 3 architecture dossier

Status: **research scaffold, no weights**. The engine recipe is registered but refuses to construct until a checkpoint
exists. Source: the Krea technical report, <https://www.krea.ai/blog/krea-3-technical-report> (published 2026-11-03).
Everything marked *inferred* is our reading, not a disclosure.

## Disclosed

- **Backbone.** Single-stream MMDiT (the report's Figure 5), described as "deliberately standard". Initialised from the
  Krea 2 pretrain and continued on native 2K data. The report's contributions are data, training and post-training.
- **Tokens.** A 1024×1024 text-to-image sample uses 4,096 image tokens.
- **References.** Each reference is VAE-encoded (the report names no VAE) and appended as clean tokens with timestep
  modulation at t = 0, which lets their K/V be cached. Up to 8 references in one generation. Reference and target
  resolution are decoupled; length-binned packing keeps native aspect ratios.
- **Text and instruction.** A VLM reads the instruction plus each reference, downsampled to a 384² pixel area. Only
  the instruction tokens reach the DiT; the VLM's vision tokens are dropped.
- **Sampling.** Rectified-flow velocity prediction with an SD3-style timestep shift. The editing shift is more
  aggressive than the text-to-image shift, value undisclosed. Classifier-free guidance is used, scale undisclosed.
- **Post-training.** RL (16 rollouts, DiffusionNFT, GDPO with a region reward) and DPO (ThermoDPO). No inference effect.
- **Variants.** Krea 3 (Fast), Ultra, Ultra Thinking. LoRA personalisation is named as a future tool.

## Inferred

- The VLM is Qwen3-VL-4B: Krea 2's text fusion is hardwired to 12 tapped layers of width 2560, which only Qwen3-VL-4B
  matches in our stack. Not stated by the report.
- The VAE is the Qwen-Image f8 VAE (16 channels) with 2×2 patches, Krea 2's geometry. It gives the disclosed token
  count.
- Layer count, hidden size, RoPE and attention shape equal Krea 2 (6144 hidden, 28 layers, GQA 12 KV heads).
- Krea 2's header carries no marker that tells the two families apart, so routing needs a metadata or filename hint.

## Undisclosed (TODO until weights)

- Parameter count, layer count and exact VLM identity and chat template.
- RoPE offsets for reference tokens, and whether text tokens sit at t = 0 as well.
- Steps, CFG scale, the editing and text-to-image shift values, default resolution.
- What the unconditional CFG pass drops (references, instruction, or both).
- VAE identity, compression factor, channel count and patch size.
- Licence and release. A co-founder said open weights are "being discussed internally". Nothing is published.
- Diffusers or ComfyUI support, and the inference stack the report promised "soon".

## Reuse map

| Need | Existing code | Krea 3 use |
|---|---|---|
| DiT blocks, attention, FFN | `Krea2Transformer`, `Krea2Block`, `Krea2Config` | Same geometry, extended with a reference path |
| Per-segment t = 0 modulation | `SegmentModulation`, `TokenSegmentLayout` (MOSS-Image) | Reference tokens at t = 0 |
| Reference preparation | `ReferenceImageResolver`, `ReferenceImagePolicy` | 384² vision copy, VAE latent copy, 8 max |
| Tapped VLM encode with images | `Qwen3VlMultimodalEncoder.EncodeMultiLayer` | Instruction plus references |
| Edit dispatch rule | `Fibo2RecipePipeline.RoutesToEdit` | RefEdit versus strength img2img |
| Reference RoPE layout | `FluxRope.BuildPositionIds(Kontext)`, `ReferencePlacement` | N references, offsets TODO |
| Scheduler | `FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift` | Edit shift as a parameter pair |
| Converter and LoRA keys | `Krea2CheckpointConverter`, `Krea2LoraMapper` | Reused until a Krea 3 key format is seen |

## Engine status

- The `krea3` recipe is registered with no image features and a prompt profile of kind `image` with a `vlm` encoder.
- Base and Fast variants are declared, with no filename tokens until a checkpoint name is known.
- A request with no checkpoint path is refused by the engine before the recipe runs (`FileNotFoundException`, "No
  checkpoint found for model 'krea3'"). Given a path, the recipe's `Construct` throws `NotSupportedException` saying no
  checkpoint is released.
- Not built: the reference DiT path, reference RoPE, the VLM edit encode, the pipeline, catalog and SwarmUI mapping.
