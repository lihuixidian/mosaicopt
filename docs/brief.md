# MosaicOpt — Design Brief
## An Algebraic Foundation for Agentic Memory Databases

> **Status**: Authoritative source for the prototype. All sub-tasks must treat this as the design contract. New design decisions must be justified.

---

## 1. Context

This document captures the design decisions for a prototype that explores an **algebraic foundation** for Agentic memory databases. The long-term goal: a prototype system that **inspires a series of research discussions** and includes a paper as part of its deliverables.

The prototype explores the question: *Can we extend relational algebra to support the multi-modal query needs of Agentic memory, and use the resulting algebra to drive principled query optimization?*

---

## 2. Background & Motivation

### 2.1 Agentic applications need memory
LLM-based agents require memory across two timescales:
- **Long-term memory**: cross-session user knowledge, episodic recall, learned facts
- **Short-term memory**: within-session context, tool call traces, intermediate reasoning state

Modern memory content is inherently **multi-modal**:
- Text (raw content, summaries, transcripts)
- Vectors (semantic embeddings, often multiple per memory)
- Structured data (metadata, tags, timestamps)
- Knowledge graphs (entity relations, derivation links)

### 2.2 Current systems lack principled foundation
Production systems (Mem0, Letta, Zep, Cognee) handle multi-modal memory by:
- Treating each modality with a separate engine (vector DB, full-text index, KV store)
- **Multi-path recall**: parallel channels, then score fusion, then rerank

This works in practice but has structural problems:
1. **No principled foundation** — query semantics are ad hoc
2. **Optimizer is heuristic** — no clear equivalence relation for plan rewriting
3. **Score fusion is forced into the substrate** — application-level policy becomes a hard architectural commitment
4. **Multi-vector columns break top-k** — independent top-k per column produces disjoint candidate sets with poor recall

---

## 3. Core Claims (the project's thesis)

1. **Top-k is too narrow for multi-modal queries.** When memory units have multiple vector columns, independent top-k per column fails to express conjunctive semantic intent. Range query (distance-bounded set retrieval) + SQL-style `ORDER BY` + `LIMIT K` is more general and substrate-appropriate.

2. **Score fusion is an application concern, not a substrate concern.** The substrate should expose per-column distance values; applications compose them via SQL `ORDER BY` expressions (`ORDER BY (d_a + d_b)/3`, `ORDER BY GREATEST(d_a, d_b, d_c)`, etc.). The substrate does not bake in fusion policy.

3. **Algebra must come before optimizer.** Without equational theory, all optimization is heuristic. Algebra provides the equivalence relation that RBO needs to rewrite plans.

4. **Relational algebra is insufficient but extensible.** It needs extension with: vector distance predicates, text relevance predicates, graph reachability predicates, and a **materialization operator** for computed values (so distance/score values can be referenced by `ORDER BY`).

5. **Logical/physical separation is essential.** HNSW is a *physical access path*, not a *logical operator*. Equivalence rules at the logical level may not hold at the physical level due to approximation. This separation is what makes the system composable.

---

## 4. Algebra Design

### 4.1 Logical Algebra (extension of relational algebra)

#### Operators
| Operator | Syntax | Meaning |
|---|---|---|
| Base relation | `R` | A stored relation |
| Selection | `σ_φ(L)` | Filter rows by predicate φ |
| Projection | `π_A(L)` | Keep only attributes in A |
| Materialization | `μ_{e→A}(L)` | Compute expression e per row, bind to fresh attribute A |
| Join | `L ⋈_ψ L` | Combine via predicate ψ |
| Set ops | `L ∪ L`, `L ∩ L`, `L − L` | Set operations on compatible relations |
| Rename | `ρ_{a→b}(L)` | Rename attribute a to b |
| Ordering | `τ_O(L)` | Order rows by expression O |
| Limit | `Λ_K(L)` | Take first K rows |
| Graph expansion | `Γ_{g, d}(L)` | From L's nodes, traverse graph rule g to depth d |

**Semantics: bag (multiset) semantics** to match SQL. Set semantics can be recovered as needed.

#### Predicates (φ)
- Classical: `A = c`, `A < c`, `A ∈ S`, etc.
- Vector: `‖V_a − q‖ < τ` where `q` is a query-time bound parameter
- Text: `BM25(t, q) > θ` where `q` is query-time bound
- Graph: `reachable(u, v)`, `edge(u, v, e)` with label `e`
- Boolean combinations: `∧`, `∨`, `¬`

#### Materialization expressions (e)
- Classical: arithmetic, function application
- Vector: `‖V_a − q‖` (distance value)
- Text: `BM25(t, q)` (relevance score)
- Boolean / arithmetic combinations

**Critical property**: materialization *exposes* computed values as first-class attributes, so they can be referenced by `ORDER BY` and subsequent `σ`.

### 4.2 Physical Algebra (brief)

Physical operators implement logical operators:
- `H_{V, q, τ}`: HNSW range scan — implements `σ_{‖V−q‖<τ}`
- `B_{col, range}`: B-tree range scan — implements `σ_{col ∈ range}`
- `F(R)`: sequential file scan
- `I_{col, term}`: inverted index scan — implements `σ_{BM25 > θ}`
- `G_{g, d, dir}`: graph adjacency traversal — implements `Γ_{g, d}`
- Enforcers: `Sort`, `TopKHeap`, `Materialize`, `Partition`, `HashBuild`, ...

**Logical-to-physical bridge**: e.g., `σ_{‖V−q‖<τ}(R)` is implementable as `H_{V, q, τ}(R)` if HNSW index exists on column V. The CBO enumerates possible physical implementations; cost model picks the best.

### 4.3 Three-Layer Architecture
```
Application layer:   SQL ORDER BY + LIMIT (policy lives here)
─────────────────────
Logical algebra:     σ, π, ⋈, μ, τ, Λ, Γ, ... (substrate contract)
Physical algebra:    H, B, F, I, G, Sort, TopKHeap, ... (engine implementations)
```

---

## 5. Key Equivalence Rules

### 5.1 Inherited (verify still hold with new operators)
- `σ_P σ_Q(R) = σ_{P∧Q}(R)` — selection composition
- `σ_P(R ∪ S) = σ_P(R) ∪ σ_P(S)` — selection distributes over union
- `π_A π_B(R) = π_A(R)` — projection idempotence
- `σ_P(R ⋈ S) = σ_P(R) ⋈ S` if P only references R — selection pushdown

### 5.2 New identities (require proof)
- **M1**: `σ_{c < τ} μ_{e→c}(R) = μ_{e→c} σ_{e < τ}(R)` where `c` is fresh — selection commutes with materialization on the materialized column
- **M2**: `μ_{e1→c1} μ_{e2→c2}(R) = μ_{e2→c2} μ_{e1→c1}(R)` where `c1, c2` fresh and `e1, e2` independent — materialization is commutative
- **M3**: `μ_{e→c} π_A(R) = π_{A ∪ {c}} μ_{e→c}(R)` where `c` fresh — materialization through projection
- **V1**: `σ_{‖V_a−q_a‖<τ_a} σ_{‖V_b−q_b‖<τ_b}(R) = σ_{‖V_b−q_b‖<τ_b} σ_{‖V_a−q_a‖<τ_a}(R)` — vector selections on distinct columns commute (correctness, not cost)
- **J1**: `σ_{‖V_a−q‖<τ}(R ⋈_ψ S) = σ_{‖V_a−q‖<τ}(R) ⋈_ψ S` where `V_a ∈ attr(R)` — vector selection pushdown
- **T1**: `τ_O Λ_K(R) = topK_O(R, K)` — limit+order equivalent to top-K (enables early termination)
- **G1**: `σ_P Γ_{g, d}(R) = Γ_{g, d} σ_P(R)` when P references only R's attributes — selection pushdown through graph expansion

### 5.3 Non-identities (counterexamples)
- **N1 (approximation)**: Logical `σ_{‖V−q‖<τ}(R)` is exactly defined. Physical `H_{V, q, τ}(R)` (HNSW) is approximate. Logical equivalence does not imply physical equivalence.
- **N2 (similarity join asymmetry)**: Logical `R ⋈_{‖V_a−V_b‖<τ} S = S ⋈_{‖V_b−V_a‖<τ} R`. Physical execution is asymmetric: HNSW on R is not the same cost as HNSW on S.
- **N3 (union + order)**: `τ_O(R ∪ S) ≠ τ_O(R) ∪ τ_O(S)` in general when R and S are bags with duplicate keys.

---

## 6. Optimizer Architecture

### 6.1 RBO (Rule-Based Optimizer)
- Operates on **logical plans**
- Applies equivalence rules to rewrite plans
- Prunes the plan space that CBO must consider
- **Output**: a set of equivalent logical plans

### 6.2 CBO (Cost-Based Optimizer)
- Operates on **physical plans**
- Enumerates physical implementations of logical plans
- Cost model: I/O, CPU, memory, (for vector) approximation parameters
- **Output**: the chosen physical plan

### 6.3 Style
Volcano/Cascades-style. Extensible through algebra: adding a new operator = adding new equivalence rules + cost model entries. No central optimizer logic change.

---

## 7. Sub-task Deliverables

### Task 1: Algebra Formalization (`docs/algebra.md`)
- **Section 3 of paper**: Extended algebra syntax (BNF) and semantics
- **Section 4 of paper**: Equational theory — axioms, derived rules with proofs, non-identities with counterexamples
- **Focus**: materialization (most novel), then vector predicates, then graph expansion
- **Length**: 4000–6000 words
- **Style**: rigorous, paper-quality, formal but readable
- **Notation**: use LaTeX-style math in Markdown (e.g., `$\sigma_{\|V-q\|<\tau}$`)

### Task 2: Presentation (`presentation/`)
- **15–20 slides** covering:
  1. Title (project name, subtitle, date)
  2. Background: Agentic memory needs
  3. Motivation case: why current systems fall short
  4. The 5 key claims
  5. Core architecture: logical/physical separation
  6. Extended algebra: new operators overview
  7. Materialization operator deep-dive
  8. Multi-modal predicates (vector / text / graph)
  9. Equivalence rules: from algebra to RBO
  10. Non-identities: when rules fail
  11. CBO: cost-based optimization layer
  12. Comparison: traditional multi-path vs. our approach
  13. Roadmap & future work
  14. Summary
- **Use** the `html-presentation-generator` skill to produce HTML slides
- **Include** speaker notes for each slide
- **Verify**: the output should render as a self-contained presentation

### Task 3: Paper Draft (`docs/paper.md`)
- **All 10 sections** of the paper structure
- **Sections 1–2**: from this brief + discussion context
- **Sections 3–5**: based on the algebra doc (and this brief if algebra doc is incomplete)
- **Sections 6–7**: optimizer architecture sketch (RBO + CBO)
- **Sections 8–9**: prototype sketch and evaluation plan (can be future-work oriented)
- **Section 10**: brief discussion
- **Length**: 6000–8000 words
- **Style**: academic paper, ready for editing/iteration
- **Format**: Markdown with LaTeX math notation

### Paper structure (10 sections)
1. Introduction & Motivation
2. Background: Relational Algebra & Modern IR
3. The Extended Algebra: Syntax & Semantics
4. Equational Theory: Axioms, Derived Rules, Proofs
5. Non-identities: When Classical Rules Fail
6. From Algebra to Optimizer: RBO Plan Generation
7. Cost Model & CBO (Physical Layer)
8. Prototype Implementation (sketch)
9. Evaluation (plan)
10. Discussion & Future Work

---

## 8. Quality Bar

- **Rigor**: this is going into a paper. Derivations must be correct, counterexamples must be valid.
- **Self-containment**: each deliverable should be readable on its own, with clear references to the brief.
- **Consistency**: terminology, notation, and design decisions must be consistent across the three deliverables.
- **No scope creep**: stick to read-only queries, single-user data model. Multi-tenancy, write paths, distributed execution are out of scope for this phase.

---

## 9. Writing Style — SIGMOD

The paper is targeting **SIGMOD** (ACM SIGMOD International Conference on Management of Data). Apply the following style conventions throughout all deliverables:

### Paper (`docs/paper.md`)
- **Structure**: Abstract → Introduction → Related Work → Preliminaries → Model/Framework → Equational Theory → Optimizer (RBO + CBO) → Implementation → Experiments → Conclusion
- **Theorem-proof style**: every non-trivial claim is stated as a `Definition`, `Theorem`, `Lemma`, or `Corollary` with a numbered proof. Use environment markers:
  ```
  **Definition 4.1** (Materialization-Commutes-with-Selection).
  ...
  *Proof.* ...
  ```
- **Formal definitions**: each operator gets a `Definition N.M` with mathematical statement, not prose.
- **Contributions list**: the introduction ends with a numbered "Our contributions" list — concise, 3–5 items.
- **Related Work**: at least one full subsection in Section 2, organized by thread (vector search, hybrid query systems, query optimization, learning-augmented indexes).
- **Density**: 12–14 pages compressed; assume the reader is a database researcher. No hand-waving, no informal "intuitively"; either prove it or call it a hypothesis.
- **Notation table**: include a Notation section (Table) at the end of Preliminaries.
- **Self-contained**: every operator and identity should be defined inside the paper. Citations point to background, not to definitions.
- **No meta-claims about "novelty"** — let the work speak. State "our position" or "we propose" without "to the best of our knowledge".
- **Math**: heavy use of LaTeX in Markdown. `$\sigma_{\|V-q\|<\tau}$`, `$\mu_{e \to c}$`, `$\Gamma_{g,d}$`, etc.

### Algebra doc (`docs/algebra.md`)
- Same theorem-proof style as the paper, but as standalone document.
- Each identity is a numbered `Theorem` or `Lemma` with `Proof`.
- Each non-identity is a numbered `Counterexample` with explicit construction.
- Include a **Summary of Identities** table at the end.

### Presentation (`presentation/`)
- SIGMOD talk style — formal but visually clean.
- 18–20 slides (slightly more than the earlier 15–20 target to allow for a Related Work slide and Contributions slide).
- Required slides added:
  - **Contributions slide** (3–5 numbered items)
  - **Related Work slide** (one overview slide summarizing the threads)
  - **Notation slide** (table of operators and predicates)
- Code examples in `monospace` font; math in MathJax-compatible LaTeX.

---

## 9. Open Questions (for the user, not for sub-tasks)

- Should the algebra support `ORDER BY` over multiple columns with the lexicographic semantic, or should we always require explicit expressions like `ORDER BY (d_a + d_b)/3`? (Lexicographic is more SQL-like; explicit expressions are more flexible.)
- Should `μ` support nested materialization (materialize a value used by another materialize expression)? My instinct is yes, but it's a design choice.
- The graph expansion `Γ` operator — should it preserve path information (return pairs `(source, target, path)`) or just the target nodes? Path-preserving is more expressive but more complex.

These can be resolved in iteration; sub-tasks should make reasonable defaults and flag them.

---

## 10. SQL Extension (Locked-in Design)

The MosaicOpt algebra is exposed to applications through a SQL extension. Four design decisions have been made (see `notes/decisions/2026-07-24-sql-extension.md` for the full discussion):

### 10.1 Predicate syntax: **SQL/MM-style** (not function-style)

Reason: function-style (`distance(emb, :q) < :tau`) reads as a UDF; SQL/MM-style is "first-class" and signals the engine commits to understanding the construct.

```sql
-- Vector range
WHERE emb IS NEAR :q1 WITHIN DISTANCE :tau1

-- Text BM25
WHERE text MATCHES :q1 WITH SCORE > :theta

-- Graph reachability
WHERE id IS REACHABLE FROM :n WITHIN 2 HOPS VIA 'derived_from'
```

### 10.2 Materialization syntax: **existing `AS`** (no new clause)

The optimizer recognizes when a `SELECT expr AS alias` value is reused in `WHERE` and `ORDER BY` and triggers M1-style rewrites automatically.

```sql
SELECT 
    emb <=> :q1 AS d1,
    emb_alt <=> :q2 AS d2
FROM memory
```

### 10.3 Graph expansion: **recursive CTE** (no new clause)

Standard SQL. The optimizer handles the recursion.

```sql
WITH RECURSIVE reachable AS (
  SELECT :n AS id, 0 AS depth
  UNION
  SELECT edge.dst, reachable.depth + 1
  FROM edge JOIN reachable ON edge.src = reachable.id
  WHERE reachable.depth < 2
)
```

### 10.4 Distance operators: **keep pgvector's `<=>`, `<->`, `<#>`**

Backward compatible with the existing pgvector ecosystem.

```sql
emb <=> :q1     -- cosine distance
emb <-> :q1     -- L2 distance
emb <#> :q1     -- inner product (negative)
```

### 10.5 Order + Limit: **no new syntax** — keep `ORDER BY ... LIMIT K`

Algebraic identity T1 governs the equivalence with top-K.

### 10.6 DDL: minimal additions

```sql
-- Vector type
col VECTOR(N)                            -- N is dimension
col VECTOR(N, METRIC := 'cosine')        -- explicit metric

-- Index DDL
CREATE INDEX idx_emb ON memory USING hnsw (emb vector_cosine_ops);
CREATE INDEX idx_text ON memory USING bm25 (text);
```

### 10.7 Resulting "two equivalent surfaces"

The same query can be written in two ways, both expressing the algebra:

**SQL surface** (what application code looks like):
```sql
WITH scored AS (
  SELECT
    id, text, ts, subsystem,
    emb <=> :q1 AS d1,
    emb_alt <=> :q2 AS d2
  FROM memory
  WHERE subsystem = 'fault-tolerance'
    AND ts > now() - interval '30 days'
)
SELECT id, text, ts
FROM scored
WHERE d1 < :tau1 AND d2 < :tau2
ORDER BY 0.7 * d1 + 0.3 * d2
LIMIT 10;
```

**Algebraic surface** (what the optimizer manipulates):
```
Q1 = Mem
Q2 = σ_{subsystem = 'fault-tolerance' ∧ ts > now() − 30d}(Q1)
Q3 = μ_{‖emb−q1‖→d1} μ_{‖emb_alt−q2‖→d2}(Q2)
Q4 = σ_{d1<τ1 ∧ d2<τ2}(Q3)
Q5 = τ_{0.7·d1 + 0.3·d2}(Q4)
Q6 = Λ_{10}(Q5)
```

The SQL parser desugars SQL into the algebra; the algebra is the optimizer's working language. The two surfaces are equivalent and one can be derived from the other (parses-to-algebra and algebra-to-SQL are well-defined functions).

---

## 11. PODS 2027 Submission Notes (Theory Track)

### 11.1 CFP facts (verified against 2027.sigmod.org/calls_papers_pods_research.shtml)

- **Document class**: `\documentclass[acmsmall,review,anonymous]{acmart}` (single-column, journal-style — *not* the SIGMOD `sigconf` class). Use acmart ≥ 1.70.
- **Length**: 15 pages + unlimited references. Appendices are unlimited in length but read at the PC's discretion.
- **Anonymisation**: double-anonymous. No author names, institutions, grant numbers, or self-identifying acknowledgements. Self-citations in third person.
- **Submission site**: https://www.easychair.org/my/conference?conf=pods2027
- **Cycles**: 1st cycle (abstract 2026-05-23, paper 2026-05-30, notification 2026-08-11); 2nd cycle (abstract 2026-12-03, paper 2026-12-10, notification 2027-03-01). The 2nd cycle gives us a more realistic runway and is the recommended target.
- **Mission reminder**: "principles of database systems … communication of new advances in the theoretical foundations of data management". PODS is theory-only; no system or experiments are required or even expected.

### 11.2 Submission package (PODS-side, frozen)

- `docs/PODS/paper.tex` — source (`acmsmall,review,anonymous`).
- `docs/PODS/paper.md` — versioned narrative companion; must stay in sync with `.tex`.
- `docs/PODS/paper.bib` — bibliography (must scrub author self-citations to third person; e.g. replace "We previously argued [Smith, 2024]" with "Smith [2024] argued").
- `docs/algebra.md` and `docs/brief.md` are NOT in the submission bundle but are required reading for the authors during the rebuttal/response window.

### 11.3 What PODS reviewers will look for

PODS evaluations focus on five things, in order of weight:

1. **Novel conceptual contribution to the *principles* of data management.** A new model, calculus, semantic framework, complexity class, or impossibility result. Not a new system, not new SQL syntax, not new benchmarks.
2. **Mathematical rigour.** Every claim stated as a theorem; every theorem proved. A "checklist of identities" without proofs is a desk-reject. The current draft has 7 identities (M1–M3, V1, J1, T1, G1) and 3 non-identities (N1–N3) — every one needs a clean proof sketch in the body, full proof in the appendix.
3. **Surprise / non-triviality.** Are the results tight (matching upper/lower bounds)? Is the new operator essential (cannot be reduced to existing ones)? Are the non-identities genuine obstructions to naive rewrites?
4. **Scope and clear limits.** What the algebra *can* and *cannot* express. The three non-identities (HNSW approximation, similarity-join asymmetry, union+order) are the paper's honest boundaries; reviewers reward authors who spell them out.
5. **Connections to existing theory.** How the algebra sits next to relational algebra, Datalog with aggregates, graph query languages, and approximation theory. The 3.7 connection to Bellantoni–Juedes-style materialization is one such hook; the physical/operator split is another.

### 11.4 Closest existing PODS work (and what makes us different)

- **Recursion / Datalog optimization**: PODS 2024 "Optimizing Nested Recursive Queries" (Temporel, Zhao et al.) treats *recursion* + non-monotone operators. We are complementary — μ is a *materialization* operator on top of the algebraic core, not a rewrite of recursion. Cite as the system-level counterpart; differentiate on logical layer + equational theory.
- **Datalog with aggregates**: Zaniolo et al. "Datalog FS" (PODS-era work) extends Datalog with frequency-support goals. We extend relational algebra (not Datalog) with multi-modal predicates and a first-class materialization operator. Different host language; different extensions.
- **Materialized views and caching**: long PODS line on view maintenance. Our μ is *intra-query* materialization under the user's control (SQL `AS`-bound), not a system-level caching decision. We share the equational flavour but the trigger is logical, not physical.
- **Graph query languages**: C2RPQ / regular queries (PODS 2017 Vardi et al.) — Γ sits in the same lineage but is a *graph-expansion* operator on relational algebra, not a graph-specific calculus. Reference the lineage; show Γ cannot be expressed as a C2RPQ because of the textual + vector pre-filters.
- **Approximate query processing**: sampling-based AQP. Our cost vector includes an `approx` component but we do not propose a new sampling estimator; instead we formalise the *contract* between logical and physical operators. Position as: we make approximation a typed property of the algebra, not a hidden detail of the engine.
- **Query rewriting and confluence**: standard RBO confluence is folklore; Newman-lemma proof for our specific rewrite system is a small but real contribution (Theorem 4.2). Treat as a clean 1-page result, not a section.

### 11.5 Differentiation summary (one paragraph reviewers will read first)

> Existing PODS work has, in isolation, studied recursive query optimisation (Temporel, 2024), Datalog extensions with aggregates (Zaniolo et al.), materialised view maintenance, regular graph query languages (Vardi et al., 2017), and approximation theory for AQP. *None* of these lines tackles the **first-class combination of text relevance, vector similarity, and graph reachability under a single logical algebra that the optimiser can reason about with provable confluence*. MosaicOpt is not a new operator; it is a *new integration principle*: the algebra is the API, and the algebra treats distance/score/reachability as first-class predicates over which the equational theory operates. The contribution is the algebra + identities + non-identities + RBO confluence proof + cost framework, viewed as a single logical system.

### 11.6 Anticipated reviewer objections and how the paper answers them

| Likely objection | Where the answer lives |
|---|---|
| "Operator μ is just materialised view substitution" | §4.4 — μ is intra-query, AS-bound, algebra-level, with explicit identities M1–M3. MV substitution is a system-level choice; μ is a logical construct the user controls. |
| "Vector predicate is a UDF in disguise" | §5 — `IS NEAR ... WITHIN DISTANCE` is a SQL/MM-style named predicate with a fixed distance function (declared via column metric) and a typed cost. UDFs are opaque to the optimiser; this is not. |
| "HNSW approximation breaks correctness" | §6.4 / Non-identity N1 — by design. The paper is explicit that σ_ann is a *physical* operator that may differ from σ; correctness is preserved only inside a declared `approx`-bounded cost envelope. |
| "The 7 identities are routine" | Appendix proof sketches are mandatory. Show that M1 depends on a *monotone* materialised column and an *equivalence* of selection ordering, not on a black-box pushdown. |
| "RBO confluence via Newman is textbook" | True, but the rewrite system is non-trivial (selection pushdown through joins, μ-through-π, materialised-σ interaction). The contribution is *applying* Newman to *this* system and showing termination, not the lemma itself. State this explicitly so reviewers don't feel cheated. |
| "Agentic memory is hype" | §2 motivates why multi-modal at this granularity needs an algebra; §8 is explicit that the algebra applies to any heterogeneous store, agentic memory is one motivation. Frame the contribution as a *principle*, not a *product*. |

### 11.7 Pre-submission checklist (PODS-specific)

- [x] All 7 identities carry full proofs in the appendix (one paragraph each is enough for 6 of 7; M1 needs a worked example).
- [x] All 3 non-identities carry a counterexample plan/instance.
- [x] Theorem 4.2 (RBO confluence) is stated with weight function, critical pair, and Newman application written out — not just cited.
- [x] No self-identifying citations, no project name in text.
- [ ] `paper.pdf` is 15 pages ± 1 with the bibliography; pages 16+ are reference + appendix only. **Current state: 18 pages total = 15 body + 1 refs + 2 appendix proofs. Body is at the 15-page limit; appendix is supplementary.**
- [ ] Cover letter (in EasyChair, not in the paper) names 3 potential PC conflicts.
- [ ] Conflicts-of-interest list filled in EasyChair at submission time.

### 11.9 Current build status (2026-07-24 17:30)

- `docs/PODS/paper.tex`: 1,650 lines, 86 KB (down from 3,117 lines, 148 KB at the start of the trim cycle — 47% line reduction).
- `docs/PODS/paper.pdf`: **18 pages** (down from 34 pages at the start of the trim cycle — 47% page reduction).
  - **Body (§1–§6)**: 15 pages (pages 1–15) — at the PODS 2027 limit.
  - **References**: 1 page (page 16).
  - **Appendix A: Proofs of Foundational Results**: 2 pages (pages 17–18) — A.1 axiom proofs, A.2 RBO Confluence proof, A.3 CBO complexity proof, A.4 cost-vector reference table.
- `docs/PODS/paper.md`: structural mirror of `paper.tex` (6 sections + appendix, 524 lines, 29 KB).
- **Section structure (10 → 6)**: §1 Introduction, §2 Extended Algebra, §3 Equational Theory, §4 SQL Surface and Rule-Based Optimization, §5 Cost Model, CBO, and Architecture, §6 Discussion.
- **HNSW → ANN**: the paper no longer assumes HNSW is the only vector retrieval path. The vector selection $\sigma_{\|V-q\|<\tau}$ is realized by an *ANN family* physical operator $H_{V,q,\tau}$ whose concrete strategy is one of {HNSW graph, IVF inverted file, brute-force linear scan}; the CBO selects per query from cost and approximation parameters. This shows up in:
  - §2.1 (background): ANN family includes HNSW, IVF, and brute-force scan.
  - §2.2 (logical/physical separation): ANN family can be added, swapped, or replaced without changing the language.
  - §4.5 (N1 counterexample): a row within $\tau$ may be missed by HNSW with small ef, by IVF with coarse nprobe, or may be missed entirely by a scan with bounded buffer — three ANN strategies, three failure modes.
  - §5.2 (DDL): `USING hnsw`, `USING ivf`, and `USING bm25` index methods all supported.
  - §5.1 (RBO): "the RBO does not decide which access path (HNSW/IVF/scan) to use".
  - §5.3 (cost model): 4-tuple cost with HNSW: $1 - \mathrm{recall}(\mathit{ef}, M)$; IVF: $1 - \mathrm{recall}(\mathit{nprobe}, \mathit{nlist})$; scan: $\mathrm{recall}=1$.
  - §5.4 (logical-physical bridge): CBO picks strategy per query; hybrid plans supported.
  - §5.5 (cost-vector table): one row each for HNSW, IVF, and brute-force scan with distinct (io, cpu, mem, approx) vectors.
- **Fig 1 (architecture diagram)**: layer rectangles are now 2.0cm tall (was 1.6cm) so the "Logical / substrate contract" and "Physical / engine implementation" labels on the left are no longer covered by the operator boxes. Per-box ANN/access-path labels are placed *inside* the layer, below the operator boxes, with HNSW/IVF/scan for the vector selection.
- **All 7 identities + 3 non-identities + RBO Confluence theorem** present in the body, with full proof sketches in the appendix.
- **Compile warnings** (none blocking): font fallback (inconsolata italic, rsfs <5.5), PDF bookmark Unicode, ~8 overfull hbox in tight inline math lines.

### 11.8 What's *not* in the PODS submission (and where it lives)

- **Prototype and experiments**: future SIGMOD submission. PODS is theory-only.
- **SQL surface details (Section 5)**: included as a *syntactic illustration*, not the contribution. Cap at 2 pages of SQL listings.
- **Implementation status, benchmarks, latency numbers**: out of scope. A single line "an accompanying prototype will be reported separately" in the introduction is fine and signals future work without violating theory-only norms.

