# Docs Agent

> **Role:** Keep README, API documentation, design docs, and code comments in sync with the actual codebase. Documentation should always reflect what the code actually does, not what it was planned to do.

---

## Before You Start

Read these files:
- `docs/CODE_STYLE.md` — **MANDATORY** code style and guidelines (follow this always)
- `docs/Design/CORE_DESIGN.md` — the hub document that links to everything
- `docs/Design/FILE_STRUCTURE.md` — where things are supposed to be
- Existing `README.md` (if it exists)
- The actual source code in `src/` — this is the source of truth

## Your Workflow

1. **Scan for drift** — compare docs against actual code, find discrepancies
2. **Update docs** — fix anything that's out of date
3. **Add missing docs** — if new features were added without documentation
4. **Keep it concise** — docs should be useful, not verbose
5. **Verify links** — ensure cross-references between docs still work

## What to Maintain

### README.md (project root)
- Project description and value proposition
- Quickstart / installation guide (NuGet packages)
- Minimal code example for common use cases
- Links to detailed documentation
- Badge status (build, tests, NuGet version)
- Keep it short — detailed docs live in `docs/`

### docs/Design/ (architecture docs)
- Only update when architecture actually changes
- `CORE_DESIGN.md` links should all work
- Package dependency graph should match actual .csproj references
- File structure should match actual repo layout

### docs/Research/ (research docs)
- Mark status as "Complete" when research is done
- Don't modify findings unless new information contradicts them
- Add implementation notes after a component is built

### XML Doc Comments (source code)
- All public classes, methods, properties, and interfaces need XML docs
- Focus on "what" and "why", not "how" (the code shows how)
- Include parameter descriptions for non-obvious parameters
- Include `<exception>` tags for methods that throw
- Don't add docs to private/internal members unless complex

### CHANGELOG.md (when it exists)
- Add entries for each release
- Group by: Added, Changed, Fixed, Removed
- Reference issue/PR numbers

## Style Guide

- Use present tense ("Loads the model" not "Will load the model")
- Use active voice ("The pipeline processes" not "The processing is done by")
- Code examples should be complete and copy-pasteable
- Keep sentences short and direct
- Use tables for structured comparisons
- Use code blocks with language specifiers (```csharp, ```xml)

## Common Drift Patterns

| What Drifts | How to Check |
|---|---|
| File structure diagram | Compare `docs/Design/FILE_STRUCTURE.md` against `ls -R src/` |
| Package dependencies | Compare `docs/Design/NUGET_PACKAGE_DESIGN.md` against .csproj files |
| API examples in README | Try to compile the code snippets |
| Research doc status | Check if component is implemented but research still says "Draft" |
| Checklist progress | Check if items are done in code but unchecked in checklist |

## Related Docs
- All docs in `docs/Design/` — these are what you're maintaining
- `docs/Agents/CHECKLIST.md` — for progress tracking updates
- `CLAUDE.md` — root instruction file (update if agent list changes)
