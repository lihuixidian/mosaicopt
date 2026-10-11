# MosaicOpt: An Algebraic Foundation for Agentic Memory Databases

> **Draft for discussion.** The algebra (Section 2) and equational theory
> (Section 3) derive from the project design brief (`docs/brief.md`) and
> track the formal algebra document (`docs/algebra.md`). All examples
> use a running Agentic memory scenario. The SQL surface is developed in
> the companion SIGMOD paper; this paper restricts itself to the
> algebraic foundation.

---

## 1. Introduction & Motivation

Large language model (LLM) agents accumulate state across two timescales:
**long-term memory** (facts, preferences, episodic traces across sessions)
and **short-term memory** (tool-call history, working context).

The content of agentic memory is fundamentally **multi-modal**: a
single memory unit may carry raw text, dense vector embeddings, structured
metadata, and graph relations (`derived_from`, `cites`, `contradicts`,
`member_of`). Useful queries are *conjunctive* across all four
modalities — a vector range intersected with a metadata filter joined
with a graph reachability predicate.

The dominant engineering response is **multi-path recall** (Mem0,
Letta, Zep, Cognee): fan-out across specialized engines, application-level
score fusion and re-ranking. The pattern has four structural weaknesses:
no principled semantics, heuristic optimization only, fusion policy
leaking into the substrate, and per-column top-k breaking conjunctive
intent.

**MosaicOpt** proposes an *algebra-first* design: extend relational
algebra with operators that make multi-modal memory queries first-class,
and let the resulting algebra — not application code — be the substrate
contract between query and execution. The optimizer is a rule-based
layer driven by an equational theory (Section 3), with a cost-based
layer (Section 4) choosing between physical realizations that may
differ in approximation.

### 1.1 Why algebraic foundation?

Codd's relational algebra and Selinger's cost-based optimizer
established that query processing benefits from a formal language and a
formal theory of what transformations preserve semantics. Volcano and
Cascades showed that the algebra is the **API**: a new operator ships
with its equivalence rules and cost model, and the optimizer gains new
capabilities without any central rewrite.

An algebra provides four things: a **language for expressive power**,
an **equational theory** (identities governing plan equivalence), a
**cost-model framework**, and **composability** (new operators can be
added without breaking old ones).

Agentic memory is the *worst case* for a "bag of engines" approach: it
spans two timescales, has genuinely multi-modal content, requires
conjunctive-across-modalities queries, and shows convergent evolution
in every framework (Mem0, Letta, Zep, Cognee).

### 1.2 Mathematical support and data substrate

A new materialization pattern can be expressed as a new operator with a
documented equational theory — rewrite rules are *derived* from the
algebra, not guessed from benchmarks. The algebra is the *shared
substrate* between application and engine: an application written once
is portable across engines that share the algebra; an engine
innovation is portable to every application that uses the affected
operator.

Four representative use cases (episodic recall, cross-session fact
retrieval, working context assembly, tool-trace analysis) are all
*query problems* amenable to the same optimizer pipeline. With
MosaicOpt's algebra, an application writes its memory query in SQL
and gets principled optimization for free; the SQL surface is the
subject of the companion SIGMOD paper.

### 1.3 Contributions

This paper makes two contributions:

- **An extended relational algebra for agentic memory** (Section 2):
  a materialization operator $\mu$, first-class vector/text/graph
  predicates, an explicit graph expansion $\Gamma$, and a strict
  logical/physical separation under bag semantics.
- **An equational theory with proofs** (Section 3): seven foundational
  identities (M1--M5, G1, G2) plus the assumed classical identities
  A1--A8 (which give the RBO the join-commutativity and
  join-associativity moves; A1 and A4 extend naturally to vector
  predicates), the derived chains and classical applications
  (M4+M1, M5+G1, A4-vector+M1, A7-symmetry, A8-regrouping), the
  $\sigma$-$\tau$ and $\tau\,\Lambda\,/\,\topK$ rewrites (both
  implied by the operator definitions), five non-identities
  (N1--N5) that approximation, graph-traversal asymmetry, or the
  traversal-vs-origin column split break; and a confluent RBO
  (Theorem 1).

The paper is intentionally foundational. The algebra is the
*shared substrate* between application and engine; a shared
vocabulary will benefit the field just as it did in the relational
world. The companion SIGMOD paper develops the implementation ---
the SQL surface, the rule-based and cost-based optimizers, and
the prototype --- that honors the algebraic contract proved here.

---

## 2. Extended Algebra

### 2.1 Background

*Relational algebra and the classical optimizer.* Codd's algebra
remains the dominant formal substrate for declarative query languages;
Selinger's cost-based System R optimizer established the template
nearly every commercial RDBMS still follows. Volcano and Cascades
generalized this to a rule-based outer loop driving a cost-based inner
search, with the algebra as the API: an equational theory of plan
identities is the foundation for principled rewriting, and
logical--physical separation allows physical innovations to be plugged
in without changing semantics.

*Vector search and modern IR.* The vector space model and BM25 are
principled relevance scoring functions; dense embeddings and ANN
indexes (HNSW, IVF, brute-force scan) make them practical at scale.
ANN indexes are approximate — the result of an HNSW range query is not
the exact set within distance $\tau$ — and this approximation is
invisible to the algebra but central to the cost model.

*Hybrid query systems.* Three lines: vector-aware RDBMS extensions
(pgvector, DuckDB vss, Oracle AI) add vector types and distance
predicates but no first-class identities; purpose-built hybrid stores
(Pinecone, Weaviate, Qdrant, Milvus, Vespa) are vector-first with
metadata bolted on; multi-modal query languages (SQL/MM, SPARQL,
Cypher) mix graph, text, and vector but rarely address all three
algebraically. Every system fuses scores at the application layer,
not by extending the algebra that defines what counts as an answer.

*The gap.* What is missing is an equational view of multi-modal memory
queries in which vector distance, BM25 relevance, and graph
reachability are first-class predicates; computed values are first-class
attributes; and the boundary between the application's fusion policy
and the substrate's evaluation strategy is contractually enforced.

### 2.2 Operator inventory and semantics

The extended algebra adds a **materialization** operator
$\mu_{e \to c}(L)$ that binds a per-row expression $e$ to a fresh
attribute $c$; first-class predicates on vector, text, and graph
attributes; and a **graph expansion** operator $\Gamma_{g, d}(L)$
that traverses a labelled-edge pattern to depth $d$ from a node-id
attribute. Bag semantics are used throughout.

A predicate $\varphi$ is a Boolean combination of classical atoms
($A = c$, $A < c$, etc.), vector atoms
($\lVert V_a - q \rVert < \tau$), text atoms
($\mathrm{BM25}(t, q) > \theta$), and graph atoms
($\mathrm{reachable}(u, v)$). Query-time bound parameters ($q$, $\tau$,
$\theta$) are first-class values, so the same algebra expression
serves all query instances.

**Remark 1 ($\Gamma$ identity at depth 0).** $\Gamma_{g,0}(L) = L$
for any rule $g$ and any plan $L$ — at depth zero no traversal
is performed, so the expansion is the identity on rows. The rule
$g$ is irrelevant at $d=0$. This is the unit of $\Gamma$
composition under the G2 transitivity rule of §3.1.

### 2.3 Example queries (Q1–Q6)

The running example: $\Mem(\mathit{id}, \mathit{text}, \mathit{ts},
\vect{emb}, \vect{emb\_alt}, \mathit{user}, \mathit{subsystem})$ and
$\KG(\mathit{src}, \mathit{dst}, \mathit{label})$. The query "find
similar passages in the fault-tolerance subsystem from the last 30
days, ranked by a weighted sum of two distances, top 10":

```
Q1 = Mem
Q2 = σ_{subsystem = 'ft' ∧ ts > now() − 30d}(Q1)
Q3 = μ_{‖emb − q1‖ → d1} μ_{‖emb_alt − q2‖ → d2}(Q2)
Q4 = σ_{d1 < τ1 ∧ d2 < τ2}(Q3)
Q5 = τ_{0.7·d1 + 0.3·d2}(Q4)
Q6 = Λ_10(Q5)
```

The fusion policy $0.7\cdot d_1 + 0.3\cdot d_2$ is application code; the
substrate has no opinion on the weights. The algebra's job is to make
this expression *rewriteable* — to compute $d_1, d_2$ only on rows that
survive the filter (M1) and to enable top-$K$ early termination
(the $\tau\,\Lambda\,/\,\topK$ recognition in Section 3.4).

---

## 3. Equational Theory

The optimizer is driven by an equational theory: identities over
logical plans, plus an explicit list of *non*-identities that
classical intuition suggests but approximation breaks.

### 3.1 Foundational axioms

The seven identities below are the *new* algebraic content
licensed by the extended operators. We do *not* state as
separate axioms the identities that are *implied* by other
operators or by classical RA: classical join commutativity (A7)
and join associativity (A8) are assumed (they are in every RA
textbook, and the RBO uses them as join-reordering moves to
enumerate the $O(n!)$ join orderings of the plan space; the
$\tau\,/\,\Lambda\,/\,\topK$ family is closed by the definition
$\topK_O(L, K) := \tau_O(\Lambda_K(L))$ from Section 2.2 and
the $\sigma$-$\tau$ commutation follows immediately from the
row-preserving semantics of $\sigma$ and $\tau$
(Definition 1). A handful of other classical identities
(selection through intersection / difference, projection through
set operations, join through set operations) are likewise
assumed. The $\Gamma$ identity at depth $0$ (Remark 1,
$\Gamma_{g,0}(L) = L$) is a definitional instance and is
mentioned but not reproved.

**M1 — Selection commutes with materialization.**
$\sigma_{c < \tau}\, \mu_{e \to c}(L) = \mu_{e \to c}\, \sigma_{e < \tau}(L)$.
For selectivity $10^{-3}$ or less, $e$ is computed on $\sim 10^3$ fewer
rows.

**M2 — Materialization is commutative.**
$\mu_{e_1 \to c_1}\, \mu_{e_2 \to c_2}(L) = \mu_{e_2 \to c_2}\, \mu_{e_1 \to c_1}(L)$.
Holds when $e_1$ and $e_2$ do not depend on each other's
intermediate outputs; the rewrite is a zero-weight $\mu$-reorder.

**M3 — Materialization through projection.**
$\mu_{e \to c}\, \pi_A(L) = \pi_{A \cup \{c\}}\, \mu_{e \to c}(L)$.

**M4 — Materialization pushdown through join.**
$\mu_{e \to c}\, (L \bowtie_\psi M) = (\mu_{e \to c}(L)) \bowtie_\psi M$
when $e$ depends only on $L$'s attributes and $c$ is fresh. This
is *not* derivable from A4/A5 (which handle $\sigma$ and $\pi$, not
$\mu$) and *not* derivable from M1/M3 (which compose $\mu$ with
$\sigma$ and $\pi$, not with $\bowtie$); it is the rewrite that
lets the Q1\u2013Q6 running example (Section 2.3) push a per-row
materialization through the similarity join.

**M5 — Materialization pushdown through graph expansion.**
$\mu_{e \to c}\, \Gamma_{g, d}(L) = \Gamma_{g, d}\, \mu_{e \to c}(L)$
when $e$ depends only on $L$'s attributes and $c$ is fresh and
is not the graph rule's node-id attribute. This is the
$\mu$-counterpart of G1 and the rewrite the RBO generates for
graph queries that materialize a per-origin attribute (e.g.\ a
distance to a fixed query point) and then expand. The two sides
produce the same multiset of $(r, v, c)$ tuples; the cost
differs because the LHS computes $e$ on every traversal row
while the RHS computes $e$ on every origin row. When
$|\Gamma_{g,d}(L)| > |L|$ (the typical case for $d \geq 1$ and
a non-trivial graph), the RHS is cheaper.

**G1 — Selection pushdown through graph expansion.**
$\sigma_\varphi\, \Gamma_{g, d}(L) = \Gamma_{g, d}\, \sigma_\varphi(L)$
when $\varphi$ references only origin attributes (not traversal
attributes). The proof relies on the row-multiplicativity of
$\Gamma$: each origin row produces one or more traversal rows, and
the multiplicity is preserved; the selection either keeps all
descendants of an origin row or drops them all.

**G2 — Graph expansion is transitive.**
$\Gamma_{g_1, d_1}\, \Gamma_{g_2, d_2}(L) = \Gamma_{g_1;g_2,\, d_1+d_2}(L)$
when $g_1, g_2$ have disjoint label sets and $g_1$'s
traversal-result attribute is disjoint from $g_2$'s source
attribute. G2 lets the RBO collapse nested $\Gamma$s into a
single expansion at depth $d_1 + d_2$ --- "two
\texttt{derived\_from} hops" becomes "one
\texttt{derived\_from} ; \texttt{derived\_from} hop". The
disjoint-label and non-overlap conditions ensure that no
intermediate $v_2$ of $g_2$ is a valid origin of $g_1$ (which
would require a join in the resulting plan and break the
single-expansion rewrite). G2 is a zero-weight rewrite bounded
by the number of nested-$\Gamma$ inversions in $P$.

### 3.2 Derived identities

Composing the foundational axioms yields rewrites the optimizer
uses directly. The chains **M4+M1** and **M5+G1** are the
non-trivial derived rewrites; **A4-vector+M1** is the
vector-specialization of **M4+M1** that the conjunctive
multi-vector case relies on; **A7-symmetry** and
**A8-regrouping** are how the RBO exposes join choices for the CBO.

- **A4-vector (in code: `J1`).** A4 specialized to a vector
  predicate \u2014
  $\sigma_{\lVert V_a - q \rVert < \tau}\, (L \bowtie_\psi M) =
  (\sigma_{\lVert V_a - q \rVert < \tau}(L)) \bowtie_\psi M$
  when $V_a \in \schema{L}$. A1 likewise specializes to two
  vector predicates on distinct columns. Neither is a new
  algebraic identity \u2014 both follow from A1/A4 by predicate
  substitution \u2014 but the codebase keeps them as named redex
  patterns (`V1`, `J1`) so the RBO can match them explicitly.
- **M4+M1 (materialization pushdown, then selection-on-materialized).**
  The rewrite that powers the Q1\u2013Q6 example of Section 2.3:
  $\sigma_{c<\tau}\,\mu_{e\to c}\,(L\bowtie M)
  \equiv \mu_{e\to c}\,(\sigma_{e<\tau}(L)\bowtie M)$ (apply M4
  then M1).
- **A4-vector+M1 (vector selection before materialization).** A
  join-side vector selection can be pushed below a materialization
  that binds the same distance for sorting: apply A4-vector then
  M1.
- **A7-symmetry (reverse join).** Applying the assumed classical
  A7 in the reverse direction is a derived rewrite the RBO
  generates; the CBO then chooses the cheaper side per N2.
- **A8-regrouping (different join parenthesization).** Applying the
  assumed classical A8 is the rewrite that exposes a different
  join parenthesization; the RBO uses it to enumerate the $O(n!)$
  join orderings of the plan space, and the CBO then picks
  the best one.
- **M5+G1 (materialization and selection through graph expansion).**
  A graph query with both a row-level filter and a row-level
  materialization can be reordered into either
  "filter-and-materialize after expand" (the input) or
  "filter-and-materialize before expand, then expand":
  $\sigma_\varphi\, \mu_{e\to c}\, \Gamma_{g,d}(L) \equiv
  \mu_{e\to c}\, \Gamma_{g,d}\, \sigma_\varphi(L)$ (apply G1 then
  M5).

The schema rules ensure that the attributes a downstream operator
references are either input attributes or attributes introduced by a
$\mu$ in the pipeline. The **$\sigma$-$\tau$ commutation** and the
**$\tau\,\Lambda\,/\,\topK$ recognition** are both
*implied-by-operator-definition* rules: the former by the
row-preserving semantics of $\sigma$ and $\tau$ (Def. 1), the
latter by $\topK_O(L, K) := \tau_O(\Lambda_K(L))$ from §2.2. The
RBO applies them as part of the zero-weight post-$w$ rewriting
pass of Section 3.4. The assumed classical identities A1–A8 plus
the seven foundational axioms plus the rewrite rules above suffice
to cover the optimization of every plan whose operators draw from
the inventory of Section 2.2.

### 3.3 Non-identities and counterexamples

**N1 — ANN approximation breaks logical equivalence.**
Logical $\sigma_{\lVert V-q \rVert < \tau}(R)$ is exactly defined;
physical $H_{V, q, \tau}(R)$ under HNSW or IVF is approximate. A
logical equivalence does not imply that two physical realizations
preserve it.
A row with $\lVert \vect{v} - \vect{q} \rVert = 0.99\tau$ may be missed
by a small-`ef` HNSW but returned by an exact linear scan.

**N2 — Similarity join asymmetry.**
$R \bowtie_{\lVert V_a - V_b \rVert < \tau} S$ and
$S \bowtie_{\lVert V_b - V_a \rVert < \tau} R$ are logically
equivalent (by A7); the two physical sides are not interchangeable
in cost.
Probing a $10^6$-row HNSW on $R$ with each row of a $10^3$-row $S$
costs $10^3$ probes; the reverse costs $10^6$ — three orders of
magnitude apart.

**N3 — Limit and order do not distribute over union.**
$\Lambda_K\, \tau_O\, (R \cup S) \neq \Lambda_K\, \tau_O(R) \cup \Lambda_K\, \tau_O(S)$.
Let $R$ and $S$ each contain 100 rows; draw $R.\mathrm{score} \sim
\mathcal{U}([0,1])$ and $S.\mathrm{score} \sim \mathcal{U}([9,10])$.
The LHS top-5 are the 5 rows of $R$ (global top by score). The RHS
gives 5 from $R$ *plus* 5 from $S$ — a 10-row result, with the
$S$ rows scoring an order of magnitude higher. The identity would
be useful for distributed top-$K$ but does not hold.

**N4 — Materialization through graph expansion fails on traversal columns.**
M5 (above) requires $e$ to depend only on *origin* attributes of
$L$. When $e$ references the traversal-result attribute produced
by $\Gamma$ (the joined-in target-node columns), $\mu$ does
*not* commute with $\Gamma$:
$\mu_{e \to c}\, \Gamma_{g, d}(L) \neq \Gamma_{g, d}\, \mu_{e \to c}(L)$.
Let $L$ have rows $(\mathit{id}, \vect{emb})$ and let
$\Gamma_{\mathrm{follows},1}$ produce an additional
$\vect{followed\_emb}$ attribute. Take
$e := \lVert \vect{followed\_emb} - \vect{q} \rVert$, a distance
to the *followed* node's embedding. The LHS materializes $c$
against the followed-node vector; the RHS first materializes
$c' = \lVert \vect{emb} - \vect{q} \rVert$ against the
*origin* vector (the followed-attribute is not yet bound) and
then expands — the two multisets compute different distances
row-by-row. N4 blocks the RBO from applying M5 when the
materialization expression reads a traversal-attribute column.

**N5 — Limit and order do not commute with graph expansion.**
$\Lambda_K\, \tau_O\, \Gamma_{g, d}(L) \neq
\Gamma_{g, d}\, \Lambda_K\, \tau_O(L)$ in general.
Let $L$ have 1000 origin rows with a $\mathit{score}$ column and
let $\Gamma_{\mathrm{cites},1}$ expand each origin to a
variable $1 \leq n_i \leq 50$ of citing nodes. The LHS
$\Lambda_{10}\, \tau_{\mathit{score}}(\Gamma(L))$ returns the 10
rows of the expansion with the globally smallest $\mathit{score}$
(some origins may contribute several winners; some origins
none). The RHS $\Gamma(\Lambda_{10}\, \tau_{\mathit{score}}(L))$
first picks the 10 origin rows with the smallest
$\mathit{score}$ and then expands each — returning up to
$10 \cdot 50 = 500$ rows that all share those 10 origins. The
two multisets differ in cardinality and in content: the LHS
asks "across all reachable nodes, which $K$ score lowest?";
the RHS asks "among the $K$ highest-scoring origins, which
nodes are reachable?". N5 blocks the RBO from pushing
$\Lambda$ or $\tau$ through $\Gamma$; the cost model of
Section 4 reflects this by giving $\tau\, \Lambda\, \Gamma$ a
substantially higher cost than $\Gamma\, \tau\, \Lambda$ when
the expansion fan-out is large.

### 3.4 Convergent rewriting

Each identity is oriented toward the optimized form, giving an
abstract rewriting system (ARS) $\to_{\mathrm{RBO}}$ on logical plans.
Define the weight $w(P) = \alpha s(P) + \beta r(P) + \gamma p(P)$
where $s$ counts $\sigma$s on $\mu$-bound attributes
(M1-fixable), $r$ counts redundant $\mu$s, and $p$ counts $\pi$s
that drop needed attributes (M3-fixable). **M1, M3, M4, G1
strictly decrease $w$**; **M2, M5, G2, and the implied $\sigma$-$\tau$
reordering are zero-weight rewrites**: M2 is a $\mu$-reordering
that preserves $s, r, p$ and is bounded by the $k!$ orderings on
$k$ materializations; M5 is a $\mu$-$\Gamma$ reordering that
preserves $s, r, p$ and is bounded by the number of
$\mu$-$\Gamma$ inversions in $P$; **G2 is a $\Gamma$-$\Gamma$
collapse** that preserves $s, r, p$ and is bounded by the
number of nested-$\Gamma$ inversions in $P$ (one collapse per
nesting); the $\sigma$-$\tau$ rewrite (implied by the
row-preserving semantics of $\sigma$ and $\tau$) preserves
$s, r, p$ and is bounded by the number of $\sigma$-$\tau$
inversions in $P$. The M1/M3/M4/G1 sub-ARS terminates by the
strict decrease of $w$; once $w$ reaches a local minimum, only
the zero-weight rewrites remain, and each terminates by its
bounded inversion count. The CBO-applied
$\topK$ recognition ($\tau_O\,\Lambda_K \leftrightarrow \topK_O$ per
$\topK_O(L, K) := \tau_O(\Lambda_K(L))$) is a definitional
rewrite that the CBO applies as a physical-realization choice;
it is in one-to-one correspondence with its inverse and so
trivially terminates. The critical
pairs from overlapping redexes
(M1$\cap$M1, M1$\cap$M4, M1$\cap$G1, M1$\cap$M5,
M4$\cap$G1, M4$\cap$M5, M5$\cap$G1, G1$\cap$A4, G2$\cap$G2,
G2$\cap$G1, G2$\cap$M5, M2$\cap$M1, M2$\cap$M5,
$\sigma$-$\tau$ $\cap$ M1, $\sigma$-$\tau$ $\cap$ M3)
all join by further rewrites, giving local confluence. (The
A7/A8 classical join-reordering identities and the
$\tau\,\Lambda\,/\,\topK$ definitional rewrite are included for
completeness; in practice the CBO applies the
$\tau\,\Lambda\,/\,\topK$ rewrite directly as a physical
realization choice, and the RBO applies A7/A8 as
join-reordering moves.) By Newman's lemma:

**Theorem 1 (RBO Confluence).** For any starting logical plan $P$ and
any two RBO rewrite sequences $P \to^*_{\mathrm{RBO}} P^*$ and
$P \to^*_{\mathrm{RBO}} P^{**}$, there exists a plan $P^\infty$ with
$P^* \to^*_{\mathrm{RBO}} P^\infty$ and $P^{**} \to^*_{\mathrm{RBO}}
P^\infty$. The normal form is unique: it depends only on $P$, not on
rewrite order.

---

## 4. Implementation: A Brief Note

The equational theory of Section 3 is the foundation on which an
implementation rests. The companion SIGMOD paper develops the full
implementation: the SQL/MM-style surface that desugars into the
algebra, the rule-based optimizer that applies the equational
rewrites, the cost-based physical-operator selector, and the
prototype built on DuckDB, `faiss-cpu`, and `rank_bm25`. The present
paper restricts itself to the algebraic content; we record here only
the design choices that *constrain* the equational theory (rather
than describe the implementation in detail).

*Logical/physical separation.* The cost model and physical operator
catalog are *separate* from the equational theory. The identities hold
for any choice of physical realization; the CBO picks a realization
from a precomputed physical catalog. The cost model is a 4-tuple
$(io, cpu, mem, approx) \in \mathbb{R}^4_{\geq 0}$, the fourth
component quantifying the recall penalty of an approximate ANN
strategy. The cost model is used to *search* the plan space, not to
*define* the equivalence class.

*TopK recognition.* The $\tau\, \Lambda$ family is closed by the
definition $\topK_O(L, K) := \tau_O(\Lambda_K(L))$; the rewrite
$\tau_O\,\Lambda_K \leftrightarrow \topK_O$ is applied by the CBO
as a physical-realization choice, not as a logical RBO rewrite. This
definitional rewriting is one-to-one with its inverse, so it
trivially terminates; the RBO's confluent ARS does not depend on it.

*SQL surface.* The SQL/MM-style surface --- `IS NEAR`, `MATCHES`,
`IS REACHABLE` --- is the application's primary interface. Its
desugaring into the algebra is total and semantics-preserving, so
every SQL query expresses a plan in the algebra, and every plan in
the algebra has a SQL rendering. The design rationale (downstream of
this paper's algebraic content) is that the SQL extension must read
as a *first-class* language construct, not as a function-style
abbreviation, so the substrate is committed to understanding the
construct and to applying the equational theory to it. The companion
SIGMOD paper develops the desugaring rules in full and reports the
experiments that validate the end-to-end pipeline.

*Boundary with the implementation.* The implementation is open-source
and is the primary artifact that the SIGMOD paper analyzes. The
equational theory in the present paper is the *contract* that the
implementation honors: every SQL query routed through the optimizer
must produce a result equal (as a multiset) to the result the algebra
evaluates, and the optimizer's rewrite sequence must terminate at a
unique normal form (Theorem 1).

---

## 5. Discussion

### 5.1 Related work

*Classical relational algebra.* MosaicOpt inherits from Codd's
relational algebra and the Selinger/Volcano/Cascades line; what
MosaicOpt adds is (i) a set of new identities (M1–M5, G1, G2) for
the extended operators, and (ii) an explicit convergent-rewriting
analysis (Section 3.4).

*Algebraic extensions of the relational model.* MosaicOpt continues a
long lineage of "extend the algebra, prove identities" work: nested
relational algebra (set-valued attributes), object-relational
extensions (user-defined types), temporal algebras (time-aware
predicates), and semi-structured algebras (tree-structured data).
MosaicOpt extends in a *new direction* — multi-modal (text + vector +
graph) in the context of agentic memory — and provides, in one
place, a SQL surface and a proven equational theory the prior
extensions did not.

*Equational theories in query optimization.* The convergent-rewriting
analysis draws on Newman's lemma, Huet's critical-pair analysis, and
the Terese textbook. The closest database prior is Atzeni and
Torlone's algebraic treatment of schema and view integration across
heterogeneous data models, whose algebraic-rewriting pattern is
identical. MosaicOpt's analysis is, to our knowledge, the first such
analysis for an algebra that includes vector, text, and graph
operators.

*Hybrid query algebra (recent).* Vector-aware RDBMS extensions
(pgvector, DuckDB vss, Oracle) add a vector type, an ANN index, and
distance predicates at the function-call level — systems, not
theories. Purpose-built hybrid stores (Pinecone, Weaviate, Qdrant,
Milvus, Vespa) are vector-first with metadata filtering bolted on.
Multi-modal query languages (SQL/MM, SPARQL, Cypher) mix graph, text,
and (in a few cases) vector, but SPARQL's vector fusion remains
open. The common pattern: every system fuses scores at the
application layer. MosaicOpt's contribution is to make the fusion
*first-class*: a materialization expression is a syntactic
construct, the optimizer sees it, the cost model accounts for it,
and the equational theory of Section 3 applies.

### 5.2 Open theoretical problems

**5.2.1 Complexity of CBO.** Classical join ordering is $O(n^2)$ for
left-deep trees but NP-hard for bushy trees. The CBO adds three new
dimensions: (i) vector selections contribute a *continuous*
(`ef`-, `nprobe`-parameterized) choice, not a discrete one;
(ii) graph expansion has a direction-parameterized realization
whose cost depends on graph density; (iii) materialization sharing
is a partition problem — each $\mu$ may be shared with another $\mu$
or kept distinct. We conjecture NP-hardness in the general case and
polynomial-time subclasses for queries that bound the number of
graph expansions, the depth of materialization sharing, and the
cardinality of physical realizations per operator.

**5.2.2 Adaptive optimization.** The cost model is static — evaluated
once at planning time — but wrong in practice (cardinality drifts,
ANN recall varies, BM25 selectivity is hard to predict). *Adaptive
optimization* must interact correctly with the algebra's convergent
rewriting: the re-planned logical plan must be a "refinement" of the
original, and any non-normal-form subplan the CBO switches to must
later re-rewrite — a property we conjecture holds locally to the
switched subtree.

**5.2.3 Distributed execution.** A distributed extension requires a
network-cost model, partition-aware physical operators (e.g. a
partition-aware $\mathrm{TopK}$ that respects N3), and an extension
of the equational theory to reason about partitioning. Graph
expansion $\Gamma$ is harder to parallelize than vector selection:
a graph expansion must traverse edges across partition boundaries,
while a vector selection is partitioned by the index itself. The
cost of $\Gamma_{g, d}$ is a function of the partition structure, not
just the graph size.

**5.2.4 Write paths and memory evolution.** The algebra is read-only.
A write algebra must model (i) incremental materialization updates
(when a base row changes, every $\mu$ that depends on it must be
invalidated or recomputed; the equational theory gives the
semantics, incremental maintenance is harder); (ii) index updates
(ANN and inverted indexes must stay consistent with the base data);
(iii) cost model drift (cardinality and selectivity parameters drift
and need re-calibration). Writes are the largest open problem.

### 5.3 Limitations and out-of-scope

- **No new physical data structures.** ANN indexes, B-trees, and
  inverted indexes are external; the contribution is the
  *algebraic* use of these primitives, not their internal design.
- **No new learning-based optimization.** CostFormer, Balsa, and the
  learned-cardinality literature are empirical contributions to the
  *physical* layer; a learned optimizer can instantiate the cost
  model without touching the algebra.
- **No new transaction model.** The algebra is read-only and
  single-node; transactional and distributed extensions are out of
  scope.

These are *scope choices* that keep the contribution focused on the
algebraic foundation.

---

## Appendix A: Proofs

### A.1 Proofs of foundational axioms

(M1) Fix $r \in L$. LHS computes $e(r)$, attaches it as $c$, and
retains $r \oplus \{c \mapsto e(r)\}$ iff $e(r) < \tau$. RHS
evaluates $e(r) < \tau$ directly: if false, $r$ is dropped; if true,
the row is extended to $r \oplus \{c \mapsto e(r)\}$ by the $\mu$.
The two multisets are equal row-by-row. The freshness of $c$
ensures the predicate $c < \tau$ in LHS is unambiguous.

(M2) For each $r \in L$, LHS computes $e_1(r)$ and then $e_2(r)$,
attaching both; RHS computes $e_2(r)$ and then $e_1(r)$. Because
$e_1$ and $e_2$ depend only on $r$ and not on each other's
intermediate result, the two sequences yield the same extension
$r \oplus \{c_1 \mapsto e_1(r), c_2 \mapsto e_2(r)\}$. The
multisets are equal.

(M3) For $r \in L$, LHS produces $\pi_A(r) \oplus \{c \mapsto
e(\pi_A(r))\}$; RHS produces $r \oplus \{c \mapsto e(r)\}$ (using
the same $e$ since $e$ depends only on $A$), then projects to
$(A \cup \{c\})$. By the support of $e$, $e(r) = e(\pi_A(r))$, and
the resulting tuples agree.

*A7 (join commutativity) and A8 (join associativity) are assumed
classical identities; their proofs are the standard textbook
one-liners and are not reproduced here.*

(M4) Fix a row $(r_L, r_M) \in L \bowtie_\psi M$ such that
$\psi(r_L, r_M)$ holds. The LHS produces
$(r_L, r_M) \oplus \{c \mapsto e(r_L)\}$. The RHS first
materializes $L$ to obtain
$\mu_{e \to c}(L) = \{r_L' \oplus \{c \mapsto e(r_L')\} : r_L' \in L\}$,
then joins with $M$ on $\psi$; since $c$ is fresh and not
referenced by $\psi$, the result is
$\{ (r_L', r_M) \oplus \{c \mapsto e(r_L')\} : r_L' \in L, r_M \in M,
\psi(r_L', r_M) \}$. Because $e$ depends only on $L$'s attributes,
the materialized value $e(r_L)$ is computed in both cases from the
same $r_L$, and $\psi$ is unaffected by the addition of $c$ to $L$'s
schema. The two multisets agree row-by-row.

(M5) For each row $r \in L$, the LHS first expands $r$ via
$\Gamma_{g,d}$ to obtain a multiset $\{(r, v_i) : v_i
\text{ reachable from } r\text{.id by } g \text{ within } d\}$
(multiplicity = number of paths), then materializes $e$ on each
traversal row. The result is the multiset $\{(r, v_i, c) : r
\in L, v_i \text{ reachable from } r, c = e(r)\}$, where $e$
depends only on $r$'s attributes by the support condition. The
RHS first materializes $e$ on $L$ to obtain
$\mu_{e \to c}(L) = \{r \oplus \{c \mapsto e(r)\} : r \in L\}$,
then expands via $\Gamma_{g,d}$. Because $c$ is fresh and is
not the graph rule's node-id attribute, the expansion
proceeds from the original node-id (unchanged by the $\mu$);
the result is $\{((r, c), v_i) : r \in L, v_i
\text{ reachable from } r\text{.id}\}$. Reordering attributes,
this is $\{(r, v_i, c) : r \in L, v_i \text{ reachable from }
r, c = e(r)\}$, the same multiset as the LHS. The two
multisets agree row-by-row.

*The vector-predicate specializations V1 and J1 are A1-vector
and A4-vector respectively; their proofs are the classical A1/A4
proofs with the vector predicate substituted in and are not
reproduced here. The $\tau\,\Lambda\,/\,\topK$ definitional
rewrite and the $\sigma$-$\tau$ commutation are both implied by
operator definitions (resp.\ $\topK_O(L, K) := \tau_O(\Lambda_K(L))$
and the row-preserving semantics of $\sigma$, $\tau$); no
separate proof is required.*

(G1) $\Gamma_{g,d}$ is row-multiplicating: each origin row
$r \in L$ produces one or more traversal rows, and the multiplicity
of $r$'s descendants is preserved. The LHS starts from $L$ and
applies $\sigma_\varphi$ *after* expansion, removing any traversal
row whose origin lies outside the satisfying set; the RHS starts
from the satisfying set and only generates traversal rows whose
origin satisfies $\varphi$. The multisets agree row-by-row, because
$\varphi$ references only origin attributes; if $\varphi$
referenced traversal attributes, the two sides would diverge.

(G2) For each $r \in L$, the LHS expands $r$ by $g_2$ to depth
$d_2$ yielding rows $(r, v_2)$, then expands each $(r, v_2)$
by $g_1$ to depth $d_1$ yielding $(r, v_2, v_1)$ for each $v_1$
reachable from $v_2$. The RHS expands $r$ by $g_1;g_2$ to depth
$d_1 + d_2$, yielding $(r, v_1)$ for each $v_1$ reachable from
$r$ by the composite. By the disjoint-label condition, $g_1$'s
traversal starts at $v_2$ and ends at $v_1$, independent of
$r$'s attributes; collapsing intermediate $v_2$ in the LHS gives
the same $(r, v_1)$ multiset as the RHS. The non-overlap
condition ($g_1$'s traversal result disjoint from $g_2$'s
source) ensures no join is needed; without it, the LHS would
require a join between $g_2$'s targets and $g_1$'s origins.

The non-identities N1–N3 are proved by counterexample in the
body of Section 3.3; the N4 and N5 counterexamples of
Section 3.3 cover the remaining cases (each shows the LHS and
RHS compute different multisets row-by-row).

### A.2 Proof of RBO Confluence

Termination holds by the weight function $w(P) = \alpha s(P) +
\beta r(P) + \gamma p(P)$ of Section 3.4: M1, M3, M4, G1 strictly
decrease $w$; M2, M5, **G2**, and the implied $\sigma$-$\tau$
reordering are zero-weight rewrites — M2 is bounded by the $k!$
reorderings on $k$ materializations, M5 by the number of
$\mu$-$\Gamma$ inversions, **G2 by the number of nested-$\Gamma$
inversions (one collapse per nesting)**, and the $\sigma$-$\tau$
reordering by the number of $\sigma$-$\tau$ inversions. The
$\tau\,\Lambda\,/\,\topK$ recognition is a definitional rewrite
that the CBO applies as a physical realization choice; it is in
one-to-one correspondence with its inverse and so trivially
terminates. A1-vector and A4-vector (V1, J1 in the codebase)
are A1/A4 applied to vector predicates and do not produce new
critical pairs beyond those of A1, A4. Local confluence holds by
the critical-pair analysis of Section 3.4: every overlapping redex
pair is joinable by a finite sequence of further rewrites. The key cases (in compact
form) are: M1∩M1 → A1+M2; M1∩M4 → M4+M1; M1∩M5 → M5+M1
($\mu$ $e$ depends on origin only); M1∩G1 → G1+M1
($\sigma$ on origin only); M4∩G1 → G1+M4 ($\mu$ $e$ on origin
only); M4∩M5 → M5+M4 (join on $L$'s side); M5∩G1 → G1+M5;
G1∩A4 → G1; **G2∩G2 → single G2 collapse (outermost redex
fires first); G2∩G1 → G1+G2 ($\sigma$ pushes through inner
$\Gamma$, then outer collapse); G2∩M5 → M5+G2 ($\mu$ moves past
inner $\Gamma$, then collapse)**; M2∩M1 → M1+M2; M2∩M5 → M5+M2;
$\sigma$-$\tau$ ∩ M1 / M3 trivially. By Newman's lemma,
termination plus local confluence implies confluence.

### A.3 Proof of CBO complexity bound

The CBO's search space is the product of the RBO's plan space and
the per-operator physical-realization choice. The RBO's plan space
is bounded by $c^n$ for a constant $c$ depending on the rule set
and $n$ operator occurrences. The CBO adds a factor of
$|\mathrm{Phys}|^n$. The product is exponential in $n$ in the
worst case. Standard pruning heuristics (Selinger-style dynamic
programming for the $\bowtie$-fragment, greedy selectivity ordering
for vector predicates, top-$K$ pruning for $\tau\, \Lambda$)
reduce the practical runtime to polynomial for the common
subclasses. The bound remains exponential in the worst case;
polynomial subclasses are characterized in Section 5.2.1.

### A.4 Reference tables

The full logical-to-physical bridge table and the cost-vector
table for core physical operators are reproduced here for
reference. The bridge table indexes each logical operator by its
realization and storage primitive (HNSW/IVF/scan for vector
selection, inverted index for BM25, B-tree for range scans,
bounded-heap top-$K$ for `TopK`, etc.). The cost-vector table
lists the symbolic cost of each physical operator as a 4-tuple
(io, cpu, mem, approx), with HNSW and IVF entries including the
$\mathrm{recall}$ term.
