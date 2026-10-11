# Humanize Diff Report — `docs/PODS/paper.md`

**Source:** `/Users/apple/CloudStation/MosaicOpt/docs/PODS/paper.md`
**Output:** `/Users/apple/CloudStation/MosaicOpt/docs/PODS/paper.humanized.md`
**Skill applied:** `humanizer:humanizer` (Wikipedia: Signs of AI writing, v2.11.2)
**Date:** 2026-09-05

---

## Final Summary

| Metric | Value |
| --- | --- |
| Total edits | **2** |
| Sections touched | 1 (§1 Introduction & Motivation) |
| Source file modified | **No** (md5 `6d2fd03f80ac1170fdcca4efa689b197` unchanged) |
| Math / citation / table / figure / heading / notation byte-identical | **Yes** (verified by `diff` of all `$…$`, `\cite`, `\ref`, `\Gamma`, `\mu`, `\sigma`, `\tau`, `\Lambda`, `\Mem`, `\KG`, and heading-prefixed lines) |
| Q1–Q6 code block byte-identical | **Yes** (verified by `diff` of code-fenced block) |

### Counts by pattern category

| Category | Count |
| --- | --- |
| §1 Inflated claims | 0 |
| §3 -ing shallow analysis | 0 |
| §4 Sales language | 0 |
| §7 Overused AI words | 0 |
| §8 `serves as` / `stands as` | 0 |
| §9 `Not X but Y` | 0 |
| §10 Forced groups of three | 0 |
| §14 Em dashes | 0 |
| §20 Chatbot artifacts | 0 |
| §23 Filler | 0 |
| §27 `fundamentally` / `at its core` | 0 |
| §32 Formulaic sayings | **2** |
| §33 Fake-candid openings | 0 |
| §35 Rejecting fake alternatives | 0 |
| Other | 0 |

### Counts by severity

| Severity | Count |
| --- | --- |
| High | 0 |
| Medium | 2 |
| Low | 0 |

**Most-edited pattern category:** §32 formulaic sayings (the only category that fired).

### Patterns explicitly NOT flagged (and why)

- **§14 Em dashes** — 35 occurrences (28 `—` and 7 spelled-out `---`) were reviewed. **All kept.** Every em dash in this paper functions as a standard PODS academic aside (parenthetical clarifications inside or alongside theorem/lemma/remark headers, list items in inline glosses, etc.). None are paired with other AI tells. Replacing them would damage the paper's existing rhythm and the theorem-label convention (`**M1 — ...**`, `**N1 — ...**`).
- **§16 Bold mini-headings in lists** — the contribution list in §1.3 and the parallel list in §2.2 use `**A language for expressive power** — a precise definition of…` style. **Kept.** This is normal PODS/SIGMOD prose.
- **§7 `fundamentally`** at line 18 — `The content of agentic memory is fundamentally **multi-modal**` — **kept.** Used in a direct, technical sense ("at a basic level / by nature"), not as the §27 "deep truth" filler. Substituting "essentially" or "inherently" would be a stylistic lateral move with no gain.
- **§7 `honors`** at lines 101 and 490 — `honors the algebraic contract` — **kept.** This is technical language from algebraic specification / contract theory (an implementation "honors" or "satisfies" a spec). Not the §7 "AI verb" usage.
- **§25 generic positive ending** at line 98 — `a shared vocabulary will benefit the field just as it did in the relational world` — **kept.** The claim is tied to a falsifiable historical observation (the relational world did benefit from a shared vocabulary), not vague optimism.
- **§15 bold text** — bold is used purposefully for theorem/lemma/remark labels (M1, M2, …, G1, G2, N1, …, Theorem 1) and contribution bullets, not as decoration. **Kept.**
- **§17 title case in headings** — instructions require headings to remain Title Case exactly as-is. **Kept.**

### Confidence call

**High.** The paper is math-dense, formal, and largely free of AI tells. The only AI pattern category that fired (§32 "the currency of") was used twice in §1 in nearly identical wording, and both instances were rewritten using vocabulary (`substrate`) that already appears elsewhere in the same paper. No fact, number, name, citation, claim, or ranking was added, removed, or rearranged.

---

## Edits, grouped by section

### §1. Introduction & Motivation

#### Edit 1 — Line 63–64

- **Pattern hit:** §32 formulaic saying ("the currency of")
- **Severity:** medium
- **Before:**
  > The algebra is the *common
  > currency* between application and engine: an application written once
- **After:**
  > The algebra is the *shared
  > substrate* between application and engine: an application written once
- **Why:** "Common currency" is on the §32 watch list (formulaic "X is the Y of Z" / "the currency of" pattern). Replaced with "shared substrate", which is a vocabulary word already established in the paper (line 34: "substrate contract", line 138: "the substrate's evaluation strategy"). Minimal, single-phrase swap; no factual change.

#### Edit 2 — Line 97

- **Pattern hit:** §32 formulaic saying ("the currency of")
- **Severity:** medium
- **Before:**
  > The paper is intentionally foundational. The algebra is the
  > *common currency* between application and engine; a shared
- **After:**
  > The paper is intentionally foundational. The algebra is the
  > *shared substrate* between application and engine; a shared
- **Why:** Same §32 pattern, same rewrite strategy as Edit 1. The two clauses surrounding the changed phrase (the "intentionally foundational" framing and the "shared vocabulary will benefit the field…" prediction) are kept intact.

---

## Self-check

1. **Did I load the humanizer skill?** Yes — `humanizer:humanizer` SKILL.md read in full (35 patterns).
2. **Did I keep all math / citations / tables / figure refs byte-identical?** Yes — verified by `diff` of all `$…$`, `\cite`, `\ref`, `\Gamma`, `\mu`, `\sigma`, `\tau`, `\Lambda`, `\Mem`, `\KG`, heading-prefixed, and table-prefixed lines between source and humanized. Zero differences.
3. **Did I add or remove any fact?** No — both edits are single-phrase substitutions. No number, name, date, citation, claim, or ranking was added, removed, or rearranged.
4. **Did I rewrite clean sentences?** No — only the two sentences containing the §32 pattern were touched. All other sentences (including all theorem/lemma/remark blocks, all proofs in Appendix A, and the Q1–Q6 code block) are byte-identical to the source.
5. **Did I write BOTH `paper.humanized.md` and `paper.humanized.diff.md`?** Yes — both files are present in `docs/PODS/`.
