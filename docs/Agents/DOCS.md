# Docs Agent

> Keep README, API docs, design docs, and code comments in sync with the codebase.

## Extra Reading
- `docs/Design/FILE_STRUCTURE.md`
- `README.md` and actual `src/` code (source of truth)

## Workflow
1. Scan for drift — compare docs vs code
2. Update stale docs; add missing docs for new features
3. Verify cross-references

## What to Maintain

**README.md:** Description, quickstart (NuGet), minimal code examples, badges. Keep short; detail in `docs/`.

**docs/Design/:** Update only when architecture changes. Ensure links work; dependency graph matches `.csproj`; file structure matches repo.

**docs/Research/:** Mark "Complete" when done. Don't modify findings unless contradicted. Add implementation notes after build.

**XML Docs:** All public APIs. Focus on "what" and "why". Include `<exception>` for throws. Skip private/internal unless complex.

**CHANGELOG.md:** Group by Added/Changed/Fixed/Removed; reference issue/PR numbers.

## Style
- Present tense, active voice
- Complete copy-pasteable code examples
- Tables for comparisons; fenced code blocks with language specifiers

## Common Drift

| Drift | Check |
|---|---|
| File structure | `FILE_STRUCTURE.md` vs actual `src/` |
| Package deps | `NUGET_PACKAGE_DESIGN.md` vs `.csproj` |
| API examples | Try to compile snippets |
| Research status | Component built but research still "Draft"? |
| Checklist | Code done but items unchecked? |
