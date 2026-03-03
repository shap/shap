---
name: ai-disclosure
description: Track and document Claude's contributions during coding sessions for PR transparency. Use when working on feature branches, PRs, or when the user wants to maintain AI contribution records. Maintains a disclosure file per branch summarizing Claude's involvement.
---

# AI Disclosure Tracking

Track Claude's contributions during coding sessions and maintain a disclosure file for PR transparency.

## Involvement Levels

1. **Autonomous** - Claude wrote the code/solution independently
2. **Assisted** - Claude implemented based on user direction
3. **Advised** - Claude provided guidance that user implemented

## File Location

**Disclosure file:** `.claude/disclosures/<branch-name>.md`

Get branch name with `git branch --show-current`. If not in a git repo, use `session-<date>`.

## Workflow

### 1. Opt-in

When starting work on a feature branch, offer once:

> "Would you like me to track AI contributions for this branch?"

If yes, create the disclosure file and begin tracking.

### 2. Track Silently

**Only track contributions made AFTER the user opts in.** Do not backfill anything from earlier in the conversation, even if it seems relevant to the branch. The disclosure file starts as a clean slate from the moment the user says yes.

After significant actions (writing functions, fixing bugs, refactoring):

1. **Log the contribution** - Append to the appropriate section in the disclosure file
2. **Record what you changed** - Track in the internal section:
   - Which files you touched
   - Brief summary of what you wrote/changed
   - Initial involvement level

Only use the disclosure file to track contributions — do not rely on in-context memory of what happened before opt-in.

No prompts during work—just silently maintain the record.

### 3. Verify and Generate Summary

When user requests summary or creates a PR:

1. **Check your work against current state:**
   - Review files you logged as touching
   - Compare current file contents against what you originally wrote
   - Use `git diff` or read the files to see if user modified them after you
   - Also recall from conversation context: did user correct you? Ask for changes? Rewrite parts?

2. **Downgrade if needed:**
   - If user significantly modified your code afterward → downgrade to Assisted
   - If user corrected your approach multiple times → downgrade to Assisted
   - Add note: "co-creation with significant user involvement"

3. **Generate the summary** with accurate involvement levels

## Disclosure File Format

```markdown
# AI Disclosure for branch: <branch-name>

## Summary
[Generated on request]

## Contributions

### Autonomous
- [One-line descriptions of independent work]

### Assisted
- [One-line descriptions of directed work]

### Advised
- [One-line descriptions of guidance provided]
```

## Internal Tracking

Track your changes in an HTML comment (not shown in final summary):

```markdown
<!--
CHANGES:
- src/partition.py: wrote repartitioning logic (autonomous)
- tests/test_partition.py: wrote validation tests (autonomous)
- src/boundaries.py: implemented boundary calc (autonomous)

CORRECTIONS:
- src/boundaries.py: user fixed off-by-one error (count: 2)
-->
```

This record lets you verify at summary time whether files still contain what you wrote, or if the user significantly changed them.

**Downgrade rule:** If user significantly modified your code or corrected your approach repeatedly, downgrade from Autonomous to Assisted and note: "co-creation with significant user involvement".

## Example Output

```markdown
# AI Disclosure for branch: feature/healpix-partitioning

## Summary
Claude assisted with repartitioning logic (co-creation with significant human involvement), autonomously wrote test cases, and advised on spatial indexing approaches.

## Contributions

### Autonomous
- Wrote test cases for HEALPix partition validation

### Assisted
- Implemented repartitioning logic based on user requirements

### Advised
- Suggested using spatial indexing for performance
```

## PR Format

When user creates a PR, offer a copy-paste block:

```markdown
## AI Disclosure

Developed with Claude assistance:
- **Autonomous**: [list]
- **Assisted**: [list]
- **Advised**: [list]

Details: `.claude/disclosures/<branch>.md`
```
