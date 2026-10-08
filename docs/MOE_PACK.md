# Expert pack

A pack stores routed experts quantized, in a directory the runtime can read as an expert source. It is the storage half of
the space reduction: the original checkpoint can be removed once `ExpertPackVerifier` passes against it.

## Layout

- `experts.bin`: one record per expert, each starting on a 4 KiB boundary. A record is three quantized projections in order:
  gate `[I, H]`, up `[I, H]`, down `[H, I]`, each in the pack's quant dtype (Q8_0, Q4_K, Q5_K, Q6_K or F32).
- `manifest.json`: format version, topology fingerprint, H, I, dtype, and per-record `(Layer, Expert, Bank, Offset, Length, SHA-256)`.
- `COMPLETE`: written last. Readers refuse a directory without it.

## Publication

Writes go to `experts.bin.partial`. `Finish` renames it to `experts.bin`, then writes the manifest, then `COMPLETE`. A crash
before `COMPLETE` leaves nothing a reader will open. A completed pack is never overwritten.

## Refusals and checks

- Opening checks `COMPLETE`, the format version, and, when given, the topology fingerprint the pack was built for.
- Every read checks the record's SHA-256. A mismatch raises `InvalidDataException` and never returns weights.
- Dimensions must be multiples of the dtype's block size (256 for Q4_K, 32 for Q8_0).

## Measured on random weights (not real checkpoints)

H = I = 256, four experts. Relative RMSE against the F32 source: Q8_0 0.38%, Q4_K 6.1%. Payload against F32: Q8_0 3.76x
smaller, Q4_K 7.1x smaller. Real checkpoints will differ; the verifier is the gate for any real pack.
