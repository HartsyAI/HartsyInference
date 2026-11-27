# Research

Use this file when a task needs upstream facts before implementation. Produce facts the implementation can use: exact
constants, shapes, byte and key layouts, reference differences, provenance and a validation plan. Take them from the
primary reference code, and pin the revision and date. Keep source snapshots separate from the current build and
verification status.

Search the existing research, modality status, PARITY_VERIFICATION and the relevant TROUBLESHOOTING entries first. Do not
duplicate facts you can recover from our code, and do not create empty outlines. Use the deep-research skill only when its
own trigger conditions are met.

A useful note names the specific reference behaviour (for example, the timestep cosine and sine order and denominator),
explains why it matters, and says how to compare saved intermediate tensors. Share the saved inputs, not just RNG seeds.
Mark unresolved questions clearly, and never turn an assumption into a claim.

Keep only the sections that apply: key numbers and layouts, any non-obvious algorithm, references, validation, and open
questions. Label dates as source snapshots. Completing the research does not establish that the engine is verified.
