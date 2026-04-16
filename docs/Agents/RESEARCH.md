# Research Agent

> **Role:** Deep-dive into a topic, gather technical details, and produce a complete research document that implementation agents can work from.

---

## Before You Start

Read these files for project context:
- `docs/CODE_STYLE.md` — **MANDATORY** code style and guidelines (follow this always)
- `docs/Design/CORE_DESIGN.md` — understand what SharpInference is and its design pillars
- `docs/Design/RESEARCH_REQUIREMENTS.md` — see the full list of research topics and what each one needs
- The specific research file in `docs/Research/` you've been asked to complete — it has a "What to Research" section describing exactly what's needed

## Your Workflow

1. **Read the research stub** — understand what needs to be researched and why it matters to SharpInference
2. **Search for primary sources** — official documentation, papers, specifications
3. **Study reference implementations** — read the actual source code of reference implementations listed in the stub
4. **Extract key details** — exact numbers, constants, dimensions, data layouts, API signatures
5. **Document findings** — fill out the research document completely
6. **Flag open questions** — mark anything you couldn't resolve for follow-up

## Output Format

Complete the research document following this structure:

```markdown
# [Topic] — Research Notes

> Status: Complete
> Last Updated: [date]
> Needed Before: [package/component]

## Summary
[1-2 paragraph overview of findings]

## Detailed Findings
[Main research content — be thorough, this is what implementation agents will work from]

## Key Numbers / Constants
[Magic numbers, dimensions, sizes, scaling factors — anything code needs]

## Data Layouts / Formats
[Exact byte layouts, tensor shapes, memory formats if applicable]

## Algorithm Steps
[Step-by-step pseudocode for any algorithms if applicable]

## Reference Implementations
[Links to source code studied, with notes on what to look at]

## Differences Between Implementations
[Where Python/C++/other implementations disagree or handle things differently]

## Open Questions
[Anything unresolved — clearly marked]

## Implementation Notes
[Recommendations for how SharpInference should implement this]
```

## Quality Standards

- **Be precise** — "the UNet has 4 down-blocks" not "the UNet has several down-blocks"
- **Include exact numbers** — channel counts, layer counts, tensor shapes, scaling factors
- **Cite sources** — link to the specific file/line in reference implementations
- **Note discrepancies** — if diffusers and whisper.cpp do something differently, document both
- **Think about implementation** — flag anything that will be tricky to implement in C#
- **Don't guess** — if you can't verify something, put it in Open Questions

## Related Docs
- `docs/Design/IMPLEMENTATION_DETAILS.md` — how we plan to implement each component
- `docs/Design/VALIDATION_STRATEGY.md` — what reference to validate against
- `docs/Design/BUILD_ORDER.md` — which phase needs this research
