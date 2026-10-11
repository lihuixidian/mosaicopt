# Humanize pass — `docs/PODS/paper.tex` → `paper.humanized.tex`

## Summary

- **Total edits:** 6
- **Severity distribution:** 0 high, 5 medium, 1 low
- **Source file unchanged:** SHA-256 `283fcf00be1aad7a58523a8714af1513ce8c8ea684cb75e22f5f0efe8fa57d42`
- **Humanized file:** `paper.humanized.tex`
- **Confidence call:** **high** — paper is already mostly clean (technical, theorem-heavy, no marketing prose). Only six surgical prose edits were warranted; everything else was either legitimate LaTeX idiom, in math, in citations, or in theorem environments.

## Counts by pattern category

| Category | Count |
|----------|------:|
| §7 AI overused words (crucial → key; impressive dropped) | 2 |
| §32 Formulaic sayings ("common currency" metaphor) | 2 |
| §27 "fundamentally" filler adverb | 1 |
| §1 Inflated / salesy phrasing ("no shortage of impressive", "most directly consequential") | 1 |

## Edits (grouped by section)

### §1 Introduction & Motivation

**Edit 1 — §7 "fundamentally" (medium)**
- **L176 (before):** `The content of agentic memory is fundamentally \emph{multi-modal}. A`
- **L176 (after):** `The content of agentic memory is inherently \emph{multi-modal}. A`
- **Why:** "fundamentally" is in the §27 filler list; "inherently" is more precise ("intrinsic to the data") and shorter.

**Edit 2 — §1 §4 salesy / "no shortage of impressive" (medium)**
- **L196 (before):** `of this pattern. It works, and there is no shortage of impressive`
- **L196 (after):** `of this pattern. It works, and product demonstrations are easy to find.`
- **Why:** "no shortage of impressive" is the sales-language "vibrant / impressive / showcases" cluster. Replaced with a neutral factual claim.

### §1.1.2 Why an algebraic foundation helps

**Edit 3 — §32 "common currency" metaphor (medium)**
- **L367 (before):** `\emph{common currency} between application and engine --- an`
- **L367 (after):** `\emph{shared basis} between application and engine --- an`
- **Why:** "common currency" is a §32 "the currency of" pattern. "shared basis" carries the same meaning without the metaphor, and is consistent with the paper's own "substrate contract" language.

### §1 end-of-introduction

**Edit 4 — §32 "common currency" (low)**
- **L418 (before):** `\emph{common currency} between application and engine; a shared`
- **L418 (after):** `\emph{shared basis} between application and engine; a common`
- **Why:** Same §32 pattern as Edit 3. Also adjusted the next clause ("a shared vocabulary" → "a common vocabulary") to avoid the doubled "shared/shared" awkwardness introduced by the edit.

### §2.1.2 Vector search and modern IR

**Edit 5 — §7 "crucial" (medium)**
- **L457 (before):** `brute-force linear scan as alternatives. A crucial property of ANN`
- **L457 (after):** `brute-force linear scan as alternatives. A key property of ANN`
- **Why:** "crucial" is on the §7 AI-overused list. "key" is shorter and equally precise in this technical context.

### §2.1.2.1 (Section 2.4 in body) Open theoretical problems intro

**Edit 6 — §1 flowery / "most directly consequential" (medium)**
- **L1158 (before):** `not yet solve. We enumerate the most directly consequential.`
- **L1158 (after):** `not yet solve. We list the most important.`
- **Why:** "most directly consequential" is a §1 inflated phrasing; "list the most important" is direct and concrete.

## Patterns explicitly NOT flagged

| Pattern | Status |
|---------|--------|
| **§14 em dashes (117 `---` occurrences)** | **Kept all.** The em dash is a legitimate LaTeX academic aside. None were paired with other AI tells. They are doing real rhetorical work (parenthetical elaborations, definitions, "X — i.e. — Y" structures). |
| **§16 Lists with bold mini-headings (e.g. `\textbf{No principled semantics.}`)** | **Kept.** Standard LaTeX enumeration style; not a tell. |
| **"pillar" / "pillars" (L438, L1097)** | **Kept.** Common academic metaphor for foundational concept, not on AI tell list. |
| **"totally ordered sequence" (L541)** | **Kept.** Mathematical term; "totally" is part of "total order". |
| **"essential for safety" (L592)** | **Kept.** Technical claim about NULL handling, not the AI-tell use of "essential". |
| **"robust" (L1305)** | **Kept.** Inside `\bibitem{}` — forbidden to touch. Also it's a paper title. |
| **"diverge" (L1506)** | **Kept.** Mathematical term in a proof; "dive" is a false-positive substring. |
| **"in order to" / "due to the fact that"** | **Not present.** |
| **"delve", "tapestry", "vibrant", "showcase", "intricate", "pivotal", "garner", "foster", "interplay"** | **Not present.** |
| **§8 "serves as / stands as / marks / represents"** | **Not present.** |
| **§9 "Not X but Y" / clipped negative endings** | **Not present in clean form.** The two "not" constructions in the paper ("algebra is the contract... not application code", "the algebraic use of these primitives, not their internal design") are genuine technical negations, not stylistic. |
| **§23 Filler "in order to" / "due to the fact that"** | **Not present.** |
| **§35 Rejecting fake alternatives** | **Not present.** |
| **§3 -ing shallow analysis ("highlighting", "underscoring", "ensuring", "reflecting", "contributing to")** | **Not present.** |
| **§10 Forced groups of three** | **Reviewed.** The 3-item enumerations found (Mem0/Letta/Zep/Cognee classification systems; "vector-aware / purpose-built / multi-modal"; A4-vector + M1, M4+M1, M5+G1 derived chains) are genuine technical lists, not stylistic. |

## LaTeX validation results

| Check | Source | Humanized | Status |
|-------|-------:|----------:|--------|
| Open brace `{` count | 882 | 882 | ✅ match |
| Close brace `}` count | 882 | 882 | ✅ match |
| `\begin` count | 45 | 45 | ✅ match |
| `\end` count | 45 | 45 | ✅ match |
| `\cite` count | 29 | 29 | ✅ match |
| `$` (math delimiter) count | 1122 | 1122 | ✅ match |
| `\begin{X}` / `\end{X}` environment pairing | OK | OK | ✅ all envs pair |
| Source SHA-256 | `283fcf00…` | (unchanged) | ✅ source byte-identical |

**pdflatex compile:** **Not run end-to-end.** The local TeX Live 2015 installation does not have `acmart.cls` available (and `tlmgr` requires admin to install it). The compile error is `! LaTeX Error: File 'acmart.cls' not found.` on line 5 — purely a missing-class issue that affects the original `paper.tex` equally, **not** a result of any humanize edit. The other structural checks above (brace balance, env-pair balance, all begin/end counts, all math delimiters, all citations) give strong evidence that the LaTeX is valid.

## Methodology notes

- The humanizer skill (`humanizer:humanizer`) is referenced in the task brief but is not installed in the local skill store. The pattern list provided in the task brief was used as the humanizer rule set.
- All edits were made via `edit` (old_string / new_string) targeting uniquely-identifiable prose runs.
- The full file was reviewed in 200-line chunks; preamble, math, citations, theorem/lemma/proof/definition/remark/algorithm environments, figure/table/lstlisting/tikzpicture environments, and bibliography were all left byte-identical.
- No new facts, numbers, names, dates, citations, or rankings were introduced.
- All section/subsection/paragraph TITLE text was preserved (Title Case).

## Out-of-scope

- `docs/SIGMOD/paper.tex` was not touched (sibling worker).
- No `.md`, `.pdf`, `.aux`, `.log`, `.out` files were modified.
- No build was run beyond the validation above.
- The preamble, math, citations, theorem environments, figure/table blocks, and lstlisting blocks remain byte-identical to the source.
