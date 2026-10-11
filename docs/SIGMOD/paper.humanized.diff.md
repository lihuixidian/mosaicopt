# Humanizer pass — `docs/SIGMOD/paper.md`

## Summary

- **Total edits:** 5
- **Severity distribution:** 1 medium, 4 low
- **Confidence call:** **high** — every edit is a 1-sentence change, no fact added or removed, no math/citation/table/figure/number perturbed.

### Counts by pattern category

| § | Pattern                                  | Hits |
|---|------------------------------------------|------|
| 7 | Overused AI word ("crucial")             | 1    |
| 9 | "Not X but Y" / clipped negative ending  | 3    |
| 32| Formulaic saying ("the X of Y")          | 1    |

### Counts by severity

| Severity | Count |
|----------|-------|
| high     | 0     |
| medium   | 1     |
| low      | 4     |

### Patterns explicitly NOT flagged (with reason)

- **§1 inflated claims** — none found. The paper uses "crucial" once (flagged separately under §7), and does not use "stands as a testament", "pivotal", "vital role", "underscores its importance", "shaping the", "evolving landscape", "marks a shift", "deeply rooted", or "key turning point" in the AI-coded sense. ("key" appears as a normal math proof word in "The key property of `μ`" — not flagged; that usage is technical, not promotional.)
- **§2 name-dropping** — none. The paper cites real references for real claims, not as proof of importance.
- **§3 -ing shallow analysis** — none in the AI sense. "that highlight specific optimizer modules" (line 4005) is a descriptive relative clause, not -ing analysis padding.
- **§4 sales language** — none. No "vibrant", "rich tapestry", "nestled", "in the heart of", "breathtaking", "renowned", "must-visit", "stunning", "showcases", "boasts", "exemplifies", "natural beauty", or "groundbreaking" in the figurative sense.
- **§5 vague sources** — none. Every claim is attributed to a specific system, paper, or section.
- **§6 formulaic challenges / outlook sections** — none. The paper has no stock "challenges" or "future outlook" sections.
- **§8 avoiding is/are ("serves as / stands as / marks / represents [a]")** — none in the AI-coded sense. "stands as a starting point for that conversation" (closing) is a normal conversational phrase, not a verb replacement.
- **§10 forced groups of three** — none flagged. Several triple-rhythms appear (e.g. "total, deterministic, and semantics-preserving" in §5.1; "the algebra is what the application writes / the algebra is what the optimizer rewrites / the algebra is what the engine realizes" in §1.3; "the same query categories, the same baselines, and the same selectivity grid" in §11.1; "the algebra is small, the rewrite rules are provable, and the optimizer is a direct consequence" in the closing). Each triple carries three independent substantive claims, not stylistic padding; flattening them would lose information.
- **§13 passive voice / missing subjects** — none excessive. The few passives are in math proofs ("the multiset is computed per row") and are natural in that context.
- **§14 em dashes** — kept all 115 → 114 occurrences (one removed in Edit 4 because the same sentence had a §9 clipped-negative "no new syntax"; the em dash was the connector for the clipped form). Remaining em dashes are standard academic asides (parenthetical inserts, list expansions, section ranges like "Sections 3—"), not paired with other AI tells.
- **§15 too much bold** — kept. Bold formatting is used in three legitimate places: (a) bold mini-headings inside numbered/bulleted lists, (b) bold emphasis on a thesis statement, (c) bold for new operator names (`μ`, `Γ`, `topK`). Each is a normal SIGMOD/PODS convention.
- **§16 bold mini-headings in lists** — kept all. The two structural lists with bold labels that the brief explicitly required (the 4 structural weaknesses in §1, the 4 algebra-provides-four-things list in §1.1) and many other bold-labelled lists (operator inventory bullets, contribution bullets, axiom labels A1–A8, M1–M5, G1–G2, N1–N5, E1–E6) are normal academic structure. Flattening them would break the paper's argument.
- **§17 title case in headings** — not changed. All section/subsection headings remain in their original Title Case ("## 1. Introduction & Motivation", "### 1.1 Why algebraic foundation?", "## 4. Equational Theory: Identities and Non-Identities", etc.) as the brief required.
- **§18 emojis** — none. No emoji decoration anywhere.
- **§19 curly quotes** — none. All quotes are straight.
- **§20 chatbot artifacts** — none. No "I hope this helps", "Let me know", "Here is", "Of course!", "Would you like", etc.
- **§21 knowledge-limit disclaimers** — none. No "as of [date]", "Up to my last training update", "not publicly available", etc.
- **§22 overly agreeable tone** — none. The paper is a formal academic voice with no chat-like praise.
- **§23 filler phrases** — none. No "in order to", "due to the fact that", "at this point in time", "has the ability to", "it is important to note that".
- **§24 too many qualifiers** — none. "It is, however, central to" and similar single qualifiers are normal academic hedging, not a stack of caveats.
- **§25 generic positive endings** — none. The closing paragraph is a concrete summary, not "The future looks bright for the field".
- **§26 too many hyphenated word pairs** — none. "high-recall", "low-recall", "real-world", "real-style", "real-workload" are used in their normal technical-adjective positions.
- **§27 pretending to reveal a deeper truth** — kept in-place. "The content of agentic memory is fundamentally multi-modal" (§1) and "The argument of this paper is simple" (closing) are not pretense — they are thesis statements the paper then justifies. Per the humanizer's "What not to flag" list, formal academic words in their natural places are not AI tells. "The algebra of Section 3 is the *optimizer's* working language; it is not what application code writes" (§5.1) is a parallel-structure correction, not a §9 clipped negative.
- **§28 announcing the next point** — none. No "Let's dive in", "Here's what you need to know", "without further ado".
- **§29 heading repeated in first sentence** — none. Section openings go straight to content.
- **§30 writing about the previous version** — none. No "this function was added to replace the previous approach" or similar.
- **§31 forced punchlines / dramatic fragments** — none. Short sentences exist only when the content is genuinely short (e.g. "Bag semantics was chosen because ..."), not as dramatic beats.
- **§33 fake-candid openings** — none. No "Honestly?", "Look,", "The thing is,", "Let's be honest".
- **§34 answering objections no one raised** — none. No "This isn't mainly about ...", "To be clear", "Some might say ... but".
- **§35 rejecting fake alternatives** — none in the AI-coded sense. Phrases like "a natural research direction" (open problems), "a natural extension" (related work), "a straightforward extension" (multi-tenancy) name real research avenues, not fake options the text dismisses.

---

## Edits, grouped by section

### Section 1. Introduction & Motivation

#### Edit 1 — line 297-299 (§32 formulaic saying; medium)

- **Pattern hit:** §32 — "X is the Y of Z" / "the currency of"
- **Severity:** medium
- **Before:** "it. This is the foundational insight: the algebra is the *common
  currency* between the application and the engine."
- **After:** "it. This is the foundational insight: the algebra is the medium
  that the application and the engine share."
- **Why:** "the common currency" is the §32 formulaic metaphor. Replaced with a plain claim that names what the algebra is shared as.

#### Edit 2 — line 307-311 (§9 "not X but Y"; low)

- **Pattern hit:** §9 — "Not X but Y" / clipped negative ending
- **Severity:** low
- **Before:** "uses the affected operator. The algebraic foundation is therefore not
  a research luxury —it is the contract that makes the database
  ecosystem composable."
- **After:** "uses the affected operator. The algebraic foundation is therefore
  the contract that makes the database ecosystem composable — not
  a research luxury."
- **Why:** "is therefore not a research luxury — it is the contract" is the "not X, it is Y" structure. Reordered to lead with the positive claim and trail the negation; the meaning and the em dash are preserved.

### Section 2. Background

#### Edit 3 — line 633 (§7 overused AI word; low)

- **Pattern hit:** §7 — overused AI word "crucial"
- **Severity:** low
- **Before:** "A crucial property of HNSW —and of most ANN indexes —is that they"
- **After:** "An important property of HNSW —and of most ANN indexes —is that they"
- **Why:** "crucial" is on the humanizer's §7 overused-AI-word list. "Important" is the natural academic equivalent and does not change the claim.

### Section 5. SQL Surface

#### Edit 4 — line 2310 (§9 clipped negative + paired em dash; low)

- **Pattern hit:** §9 — clipped negative ending; §14 — em dash removed because it carried the clip
- **Severity:** low
- **Before:** "Graph traversal is exposed as a standard SQL `WITH
  RECURSIVE` clause —no new syntax. The optimizer handles the"
- **After:** "Graph traversal is exposed as a standard SQL `WITH
  RECURSIVE` clause; the standard `WITH RECURSIVE` form is reused as-is.
  The optimizer handles the"
- **Why:** "clause —no new syntax" uses the §9 clipped form (no new syntax = we do not add new syntax). Expanded the clip to a full clause and replaced the em dash with a semicolon. The substantive claim (no new SQL keyword is introduced for graph traversal) is preserved.

### Section 11. Experimental Study

#### Edit 5 — line 3861 (§9 clipped negative; low)

- **Pattern hit:** §9 — clipped negative ending
- **Severity:** low
- **Before:** "is *executable*, not just paper."
- **After:** "is *executable*: it runs on real data, not only on paper."
- **Why:** "not just paper" is a §9 clipped ending. Expanded to a full clause: "runs on real data, not only on paper". The claim that the algebra is executable is preserved verbatim; the contrast with the paper-only view is also preserved.

---

## Self-check

1. **Did I load the humanizer skill?** Yes — called `skill(name="humanizer")` and read the full SKILL.md.
2. **Math / citations / tables / figure refs byte-identical?** Verified: inline `$...$` math unchanged (11 → 11), display `$$...$$` unchanged (12 → 12), bracket-style citations unchanged (9 → 9), table rows unchanged (103 → 103), code fences unchanged (38 → 38), theorem/axiom labels (M1–M5, G1–G2, N1–N5, A1–A8) unchanged (144 → 144), bold labels unchanged (154 → 154). Em dashes changed by 1 (115 → 114) — that is the em dash removed in Edit 4, which was paired with the §9 clipped "no new syntax".
3. **Did I add or remove any fact?** No. Each rewrite preserves every number, name, claim, and reference. Edit 1 changes "common currency between A and E" to "medium that A and E share" (same fact, different metaphor). Edit 2 reorders the existing X and Y. Edit 3 swaps one adjective for its synonym. Edit 4 expands a clipped clause into a full one with the same meaning. Edit 5 expands a clipped ending into a full one with the same meaning.
4. **Did I rewrite clean sentences?** No. The other ~4500 lines of the paper were not touched.
5. **Both output files written?** Yes — `docs/SIGMOD/paper.humanized.md` and `docs/SIGMOD/paper.humanized.diff.md`.
6. **Source untouched?** Yes — SHA-256 of `docs/SIGMOD/paper.md` is `9386b7d83d10bad20971928757a8533ebbe8252e0f4a0f543dac62dae4bcaa5e` before and after.
