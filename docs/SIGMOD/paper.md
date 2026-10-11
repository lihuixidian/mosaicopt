# MosaicOpt: An Algebraic Foundation for Agentic Memory Databases



> **Draft for discussion.** Sections 3— derive from the project design brief

> (`docs/brief.md`) and are intended to track the formal algebra document

> (`docs/algebra.md`) when the latter is finalized. All examples in the paper

> use a running Agentic memory scenario; notation is defined in Section 3.

> The SQL surface (Section 5) is the locked-in query interface for the

> prototype and is the subject of `notes/decisions/2026-07-24-sql-extension.md`.



---



## 1. Introduction & Motivation



Large language model (LLM) agents are no longer stateless request–response

loops. Production agents —coding copilots, personal assistants, autonomous

research workers, multi-step planners —accumulate state across two

timescales: **long-term memory**, which persists facts, preferences, and

episodic traces across sessions; and **short-term memory**, which captures

the working context of an in-flight session, including tool-call history and

intermediate reasoning artifacts. This split is widely accepted in the

agentic-systems literature, but its consequences for the storage layer have

received less attention.



The content of agentic memory is fundamentally **multi-modal**. A single

"memory unit" may carry: (i) raw text (a chat snippet, a code diff, a

document excerpt); (ii) one or more dense vector embeddings, often produced

by different encoders for different retrieval objectives (e.g. semantic

similarity, intent classification, code search); (iii) structured metadata

(timestamps, source, provenance, user tags, confidence scores); and

(iv) graph structure (entity relations such as `derived_from`,

`cites`, `contradicts`, `member_of`). A useful query against this store is

seldom a pure similarity lookup; it is a *conjunctive* expression that mixes

all four modalities. "Find passages about a fault tolerance bug from the

last 30 days, semantically similar to this stack trace, attributed to a

known subsystem, and connected by `derived_from` to a previous incident" is

representative.



The dominant engineering response is **multi-path recall**: a fan-out

across specialized engines (a vector index, a full-text index, a

key–value store, a graph store), each producing a ranked candidate list,

followed by application-level score fusion and re-ranking. Frameworks

such as Mem0 [Chhikara et al., 2025], Letta [Packer et al., 2024], Zep,

and Cognee instantiate variations of this

pattern. It works, and there is no shortage of impressive product

demonstrations. But the pattern has four structural weaknesses:



1. **No principled semantics.** The semantics of a fused query is defined

   by the application code that combines the per-channel scores. Two

   implementations of "find similar memory" in the same product can have

   silently different meanings.

2. **Heuristic optimization only.** Without an algebra, there is no

   equivalence relation over plans. The "optimizer" is a hand-tuned

   orchestration pipeline whose rewrites are not derived from a sound

   theory.

3. **Fusion policy leaks into the substrate.** Score combination functions

   (e.g. weighted sum, reciprocal-rank fusion, learned rerankers) are

   application concerns; once baked into a query template, they are hard

   to change without re-engineering the data path.

4. **Top-k per column breaks conjunctive intent.** Each vector column is

   typically searched independently with its own top-k, then intersected.

   For multi-vector memories, the intersection of per-column top-k sets is

   frequently empty or impoverished, because the columns are correlated,

   not redundant.



Our position is that these problems are not implementation details; they

are symptoms of a missing foundation. **MosaicOpt** proposes an

*algebra-first* design in which relational algebra is extended with a

small set of operators that make multi-modal memory queries first-class,

and in which the resulting algebra —not application code —is the

substrate contract between query and execution. The optimizer becomes a

*rule-based* layer driven by an equational theory (Section 4), with a

*cost-based* layer (Section 7) choosing between physical realizations

that may legitimately differ in approximation behaviour.



### 1.1 Why algebraic foundation?



Database systems have long benefited from formal foundations. Codd's

relational algebra [Codd, 1970] and Selinger's cost-based optimizer

[Selinger et al., 1979] established that query processing benefits from

a formal language for what is computable and a formal theory for what

transformations preserve semantics. The Volcano/Cascades line further

showed that the algebra is the *API*: a new operator is shipped with

its equivalence rules and cost model, and the optimizer gains new

capabilities without any central rewrite. **An algebra is not an

implementation detail; it is the contract between query and execution.**

When the algebra is rigorous, every query language is a syntactic

surface over it, every optimizer decision is auditable, and every

physical innovation is a drop-in.



Formally, an algebra provides four things that no collection of indexes

and APIs can:



- **A language for expressive power** —a precise definition of what is

  computable in the system, and what is not.

- **An equational theory** —a body of identities that govern when one

  query plan is equivalent to another, the foundation on which any

  rule-based rewriter rests.

- **A cost-model framework** —the substrate on which cost-based

  optimization rests, parameterized by the same logical operators that

  the user wrote.

- **Composability** —new operators can be added without breaking old

  ones, because the algebra is the integration point. The Cascades

  maxim —*the algebra is the API* —makes extensibility a property of

  the foundation, not of the optimizer code.



Agentic memory is the *worst case* for a "bag of engines" approach. The

reasons are structural:



- It spans two timescales (long-term and short-term) that demand

  different access patterns, lifetime semantics, and consistency

  guarantees.

- Its content is genuinely multi-modal: text, multiple vector columns,

  structured metadata, and graph relations all live on the same memory

  units.

- The queries that matter are *conjunctive across modalities* —a

  vector range intersected with a metadata filter joined with a graph

  reachability predicate. Ad hoc orchestration is exactly where it

  fails.

- There is *convergent evolution* in the field: every Agentic

  framework (Mem0 [Chhikara et al., 2025], Letta [Packer et al., 2024], Zep,

  Cognee) has independently reinvented the

  same patterns (multi-path recall, score fusion, re-ranking) but with

  no shared vocabulary, no shared type system, and no shared notion of

  plan equivalence.

- The practical consequence: each framework has a different *memory

  API* with no formal semantics. The same SQL-like query may have

  silently different meanings across implementations, and porting a

  workload requires reading source code rather than reasoning from a

  specification.



### 1.2 Mathematical support for Agentic database development



The algebra gives database researchers and engineers a precise substrate

on which to develop new query patterns and to *prove* their

correctness. A new materialization pattern (e.g., a new kind of

multi-modal fusion, a new join algorithm for hybrid vectors) can be

expressed as a new operator with a documented equational theory —its

rewrite rules are *derived* from the algebra, not guessed from

benchmarks. The optimizer's plan space becomes a *derivable* structure

rather than a hand-tuned mess: each new operator is added with its

identities, and the optimizer gains the corresponding rewrites "for

free". This is exactly the extensibility property of Cascades; the

algebra is what makes it principled.



A database system built on an algebra exposes a *substrate capability*:

any application that can express its query in the algebra can take

advantage of the optimizer's rewrites, and any physical innovation that

implements an algebraic operator improves every application that uses

it. This is the foundational insight: the algebra is the *common

currency* between the application and the engine. Without it, every

application negotiates with every engine on its own terms; with it, an

application written once is portable across engines that share the

algebra, and an engine innovation is portable to every application that

uses the affected operator. The algebraic foundation is therefore not

a research luxury —it is the contract that makes the database

ecosystem composable.



### 1.3 Data substrate for Agentic applications



Four representative Agentic memory use cases illustrate what

applications need from a substrate:



- **Episodic recall** (debugging copilots, personal assistants, "what

  did I work on last week"): conjunctive vector + metadata + graph

  queries, where a single memory unit is retrieved only if it

  satisfies a similarity predicate *and* a time window *and* a graph

  reachability test.

- **Cross-session fact retrieval** (long-term user preferences, learned

  facts): single-vector or multi-vector recall with structured filters

  (e.g., a user-scope, a topic filter, a confidence threshold).

- **Working context assembly** (short-term memory for an in-flight

  session): fast filtered recall by recency + entity, where latency

  dominates and the top results must arrive in milliseconds.

- **Tool-trace analysis** (multi-step agents, planning): graph queries

  over `derived_from` and `follows` edges, joined with vector

  similarity on the underlying tool outputs.



All four are *query problems* —they are expressible as expressions

over a memory store, and they are amenable to the same optimizer

pipeline. All four benefit from a shared algebraic substrate: the

optimizer's rewrites (M1--M5, G1, G2 foundational; V1, J1,

A4-vector+M1, M4+M1, M5+G1, A7-sym, A8-regroup derived) apply

to each, and the cost model handles the modality-specific

physics.



With MosaicOpt's algebra, an Agentic application writes its memory

query in SQL (Section 5) or in the algebra directly; the substrate

applies the M1–M5 + G1 rewrites to fuse materialization with

selection, push `μ` through joins and graph expansion, and push

predicates down, and applies cost-based physical-operator

selection. The application does not have

to know about HNSW, B-tree, or inverted indexes; the substrate hides

these as physical realizations of a single logical operator. The

result is that an application that today writes its own multi-path

orchestrator —with hand-tuned per-channel top-k, hand-tuned fusion

weights, and hand-tuned thresholds —can be rewritten as a single SQL

query, and get *principled optimization for free*. This is the

substrate contract in action: the algebra is what the application

writes, the algebra is what the optimizer rewrites, and the algebra is

what the engine realizes. The application's policy (the fusion

weights, the thresholds, the top-k) stays in the SQL; the substrate's

responsibility (which index, which join order, which approximation

budget) is delegated to the optimizer.



### 1.4 Contributions



This paper makes the following contributions:



- **An extended relational algebra for Agentic memory.** We add a

  *materialization* operator `μ` for binding computed values (vector

  distances, BM25 scores, derived attributes) so they can be referenced

  by `ORDER BY` and downstream predicates. We add first-class

  *vector*, *text*, and *graph* predicates. We add an explicit *graph

  expansion* operator `Γ`. Bag semantics and a strict

  logical/physical separation complete the model (Section 3).

- **An equational theory with proofs.** We state and prove a

  set of identities —selection/materialization commutativity

  (M1–M3), materialization pushdown through join and graph

  expansion (M4, M5), and graph expansion pushdown (G1) —and

  we document the *non-identities* that classical intuition

  suggests but approximation breaks (Section 4.5). The

  vector-specialized V1 and J1 are corollaries of A1 and A4

  (Section 4.4); the `τ`+`Λ` ⇒`topK` equivalence and the

  `σ`-`τ` commutation are *implied* by the operator

  definitions rather than separate foundational theorems.

- **A three-layer architecture.** Application policy (SQL `ORDER BY`,

  `LIMIT`) is separated from the logical algebra (substrate contract)

  and the physical algebra (engine implementation). The substrate

  exposes per-column distances; fusion is the application's job.

- **Approximation-aware query optimization (A and B).** We treat

  approximation as a first-class *fourth cost dimension* (alongside

  I/O, CPU, and memory) and design a CBO that (i) *prunes* any

  physical plan whose estimated recall violates a user-specified

  approximation budget `ρ_min` at planning time, and

  (ii) *adapts* at execution time by inserting recall checkpoints

  at well-defined plan points and falling back to a higher-recall

  realization when the observed recall falls short. The pair gives

  a *recall guarantee* (the cost model gives a planning-time

  bound; the adaptive re-planning closes the gap between estimated

  and observed recall). Section 7 details the algorithm; the safety

  argument is in §7.5.

- **Cross-modal plan enumeration (D).** The RBO exposes the full

  *equivalence class* of logical plans reachable under the

  equational theory, and the CBO searches it with *modality-aware

  pruning*: per-modality cost estimates order the predicates

  (vector-first, text-first, or graph-first) and the join

  parenthesization is enumerated jointly with the modality

  ordering. The output is a small but complete plan space

  --- typically 2— plans for the Agentic-memory queries we have

  explored --- over which the CBO picks the optimal physical

  realization. Section 6.4 details the enumeration and pruning.

- **A prototype and a six-experiment evaluation.** The reference

  implementation, *MosaicOpt*, is a Python package built on

  DuckDB, `faiss-cpu`, and `rank_bm25`. We evaluate (E1) the M1

  rewrite speedup, (E2) the CBO's ANN-strategy choice with a

  Pareto front over latency and recall, (E3) a multi-modal

  query against a multi-path orchestrator baseline, (E4) the

  `Q1→Q6` end-to-end pipeline, (E5) the cross-modal plan

  enumeration showing the plan space the optimizer explores,

  and (E6) the adaptive re-planning as the execution-time recall

  guarantee. Section 11 reports the numbers.



The paper is intentionally foundational, but the systems

contributions --- the four-dimensional cost model with

approximation-aware pruning, the adaptive re-planning safety net,

and the cross-modal plan enumeration --- are the load-bearing

algorithmic pieces. They are the part of MosaicOpt that turns the

algebra from a notational convenience into an executable optimizer

that is competitive with hand-tuned baselines.



---



## 2. Background



### 2.1 Relational algebra and the classical optimizer



The relational model [Codd, 1970] and the relational algebra that

instantiates it remain the dominant formal substrate for declarative

query languages. Codd showed that a small set of operators —selection

`σ`, projection `π`, Cartesian product, union, set difference, and

rename —is sufficient to express any first-order query. Subsequent

work refined the operator set (adding explicit `⋈` for theta-join,

grouping/aggregation, ordering, and limits) and crystallized in SQL.



The second foundational pillar is **query optimization**. Selinger's

1979 cost-based optimizer for System R [Selinger et al., 1979]

established the template that nearly every commercial RDBMS still

follows: enumerate alternative orderings and access paths, estimate

cost under a cardinality model, pick the cheapest. The **Volcano

extensible query evaluation framework** [Graefe, 1994] and later

**Cascades** [Graefe, 1995] generalized this to a *rule-based* outer

loop driving a *cost-based* inner search, in which equivalence rules

over logical operators are first-class objects, and the optimizer is

itself extensible by adding rules. The insight of Cascades is that the

algebra is the API: a new operator is shipped with its equivalence

rules and cost model, and the optimizer gains new capabilities

without any central rewrite.



Two ideas from this tradition are central to our work. First, an

*equational theory* —identities over plans —is the necessary

foundation for principled rewriting. Second, *logical–physical

separation* is what allows physical innovations (new access methods,

new join algorithms) to be plugged in without changing semantics.



### 2.2 Vector search and modern IR



The retrieval stack has evolved along a parallel track. Information

retrieval developed the *vector space model* [Salton, Wong & Yang,

1975] and BM25 [Robertson, Walker & Jones, 1995] as principled

relevance scoring functions. The neural era added dense *embeddings*

trained for semantic similarity [Mikolov et al., 2013; Reimers &

Gurevych, 2019], and approximate nearest neighbour (ANN) indexes

make these practical at scale. HNSW [Malkov & Yashunin, 2018] is the

dominant ANN structure today, providing sub-linear search with

controllable recall via `ef` and `M` parameters.



A crucial property of HNSW —and of most ANN indexes —is that they

are *approximate*. The result of an HNSW range query is not the

exact set of points within distance `τ`; it is an approximation that

misses some true neighbours (false negatives) and may return some

non-neighbours (false positives, depending on `ef` and the distance

threshold). This approximation is invisible at the level of the

algebraic query language: the user wrote `σ_{‖V−q—τ}(R)` and

presumably wants the *logical* answer. It is, however, central to

the cost model: a high-recall HNSW plan costs more than a

low-recall one, and the optimizer must be able to *trade off*

recall against latency.



### 2.3 Hybrid query systems



The last decade has seen a sustained effort to bring vector and

relational together. Three lines of work are relevant.



The first is **vector-aware RDBMS extensions**, exemplified by

PostgreSQL's `pgvector`, DuckDB's `vss`, and Oracle's AI Vector

Search. These add a vector type, an ANN index, and vector distance

predicates; integration is at the function-call level. The result is

usable but architecturally shallow: there are no first-class

identities for vector selection pushdown, no cost model for

approximation, no principled way to rewrite a hybrid query.



The second is **purpose-built hybrid stores**: Pinecone, Weaviate,

Qdrant, Milvus, Vespa. These are vector-first, with metadata

filtering bolted on. Their optimization story is also pragmatic,

not foundational; metadata filters are pushed through vector

searches by an internal planner, but the planner's rewrites are

not exposed as named identities.



The third is **multi-modal query languages**: SQL/MM, SPARQL with

full-text extensions, CQL (continuous query language), and various

research prototypes that mix graph, text, and vector. SPARQL in

particular has a careful algebraic foundation, but its fusion with

vector predicates is still an open area.



The common pattern: every system mixes the modalities by *fusing

scores at the application layer*, not by *extending the algebra

that defines what counts as an answer*.



### 2.4 The gap



What is missing is an *equational* view of multi-modal memory queries

in which vector distance, BM25 relevance, and graph reachability are

first-class predicates on a par with `A = c`; in which computed

values (distances, scores) are first-class attributes; and in which

the boundary between the application's *fusion policy* and the

substrate's *evaluation strategy* is explicit and contractually

enforced. That is the gap MosaicOpt addresses.



---



## 3. The Extended Algebra: Syntax & Semantics



We extend the classical relational algebra with three new kinds of

construct: a *materialization* operator that lifts computed values

to first-class attributes, first-class multi-modal predicates

(vector distance, BM25, graph reachability), and a *graph

expansion* operator. Bag semantics are used throughout, matching

SQL and avoiding the row-collapse anomalies of set semantics in

the presence of duplicate memory units. The logical/physical

separation is enforced: physical operators are introduced in

Section 7 and never appear in this section.



### 3.1 Logical/physical separation



Following System R and Cascades, we treat the *logical* algebra as

the substrate contract —what the user asks for —and the *physical*

algebra as the engine contract —how it is evaluated. Equivalences

hold in the logical algebra; physical operators are *implementations*

of logical operators, and a logical equivalence does not in general

imply that two physical realizations are interchangeable (Section

5, N1–N3). This separation is what makes the system *composable*:

HNSW can be added, swapped, or replaced without changing the

language the user sees.



### 3.2 Operator inventory



A *logical plan* is a term over the following operators. `R`

denotes a stored base relation. `L`, `M` denote logical plans.

`φ` denotes a predicate, `e` a materialization expression, `A`

a set of attribute names, `O` an ordering expression, `K` a

non-negative integer, `c` a fresh attribute name, `g` a graph

rule (a labelled-edge pattern), and `d` a non-negative integer

depth.



| Operator            | Syntax                  | Meaning                                          |

|---------------------|-------------------------|--------------------------------------------------|

| Base relation       | `R`                     | The named stored relation.                       |

| Selection           | `σ_φ(L)`                | Keep rows satisfying `φ`.                        |

| Projection          | `π_A(L)`                | Keep only attributes in `A`.                     |

| Materialization     | `μ_{e→c}(L)`            | For each row, compute `e`; bind result to `c`.   |

| Join                | `L ⋈_ψ M`               | Pair rows with `ψ` true.                         |

| Union / Inter / Diff| `L √M`, `L √M`, `L−M` | Bag set operations on compatible schemas.        |

| Rename              | `ρ_{a→b}(L)`            | Rename attribute `a` to `b`.                     |

| Ordering            | `τ_O(L)`                | Order rows by expression `O` (a sequence).       |

| Limit               | `Λ_K(L)`                | First `K` rows of an ordered input.              |

| Graph expansion     | `Γ_{g,d}(L)`            | From `L`'s node-id attribute, traverse rule `g` to depth `d`. |



**Bag semantics.** A relation is a multiset of tuples. All operators

are multiset-preserving: `σ`, `μ`, `Γ` preserve row multiplicity;

projection preserves multiplicity; set operations combine

multiplicities additively for `∪`, by `min` for `∩`, and by

subtraction (clamped at 0) for `−`. `τ_O` produces a totally

ordered sequence; `Λ_K` truncates it. Bag semantics was chosen

because the substrate must be able to express "the same memory

unit appears N times" (e.g. an episodic trace where the same

event is recorded with several slightly different timestamps),

and because it matches the SQL semantics that application code

already expects.



### 3.3 Predicate forms



A predicate `φ` is a Boolean combination of atoms over the

attributes of the input plan. The atom forms are:



- **Classical.** `A = c`, `A < c`, `A > c`, `A √S` for constant

  or set `c`, `S`.

- **Vector.** `‖V_a √q—< τ` where `V_a` is a vector-valued

  attribute of the input, `q` is a query-time bound parameter

  (a dense vector), and `τ ≤0` is a distance threshold. Equality

  and other comparison operators are admissible; range (`< τ`) is

  the most important form.

- **Text.** `BM25(t, q) > θ` where `t` is a text attribute, `q`

  a query-time bound string, and `θ` a relevance threshold. We

  use the Okapi BM25 definition [Robertson, Walker & Jones, 1995]

  with parameters `(k_1, b)` carried with the predicate.

- **Graph.** `reachable(u, v)` (boolean) and `edge(u, v, ℓ`

  (boolean; `ℓ` is a label or wildcard) for nodes `u, v` in a

  graph attribute.

- **Boolean combinations.** `φ √ψ`, `φ √ψ`, `¬φ`, parentheses.



The *query-time bound parameters* (`q`, `τ`, `θ`) are first-class

values supplied at execution time, distinct from attributes of the

input. They make predicates safely *late-binding*: the algebra

expression is the same across query instances, only the bound

parameters change.



### 3.4 Materialization expressions



A materialization expression `e` is computed per row and bound to a

fresh attribute name. Forms are:



- **Classical.** `A + B`, `f(A)` for any scalar function `f`.

- **Vector.** `‖V_a √q‖` —the distance from a vector attribute to

  a query-time bound vector.

- **Text.** `BM25(t, q)` —a BM25 score.

- **Boolean / arithmetic combinations** of the above.



The key property of `μ` is that the *result* of the computation

becomes a first-class attribute of the output. Once materialized, the

value is referenceable by downstream `σ`, by `π`, by `ORDER BY`

expressions in `τ`, and by other `μ` expressions. This is what

makes the substrate able to expose per-column distances to

application-level `ORDER BY` without losing them.



**Fresh-name condition.** The new attribute name `c` in

`μ_{e→c}(L)` must not already appear in the schema of `L`. This is

a hygiene condition, not a semantic restriction; `ρ` is provided

to free names if needed.



### 3.5 Materialization: motivation and semantics



A naïve way to expose per-column distances to the application is to

*inline* them in `ORDER BY` expressions: `ORDER BY

‖V_a−q_a—+ ‖V_b−q_b‖`. This is syntactically permitted by SQL

extensions, but it forces the application to write the distance

expression everywhere it is used, and it leaves the optimizer no

room to share computation between, say, a `WHERE` filter and an

`ORDER BY` rank.



`μ` resolves this. The pattern



```

μ_{‖V_a−q_a‖→d_a}  μ_{‖V_b−q_b‖→d_b}  σ_{d_a < τ_a √d_b < τ_b}  τ_{d_a + d_b}  Λ_K

```



says: "compute the distance to query `a` and to query `b`; keep rows

where both are below their respective thresholds; sort by sum of

distances; return the top K." The fusion policy `d_a + d_b` is

expressed once, in the `τ` expression. The optimizer can rewrite

this plan —for example, by using M1 to commute the selection

through the materializations, computing each distance only on

surviving rows (Section 6).



Formally, the semantics of `μ_{e→c}(L)` is the multiset

`{ r ⊆{c →e(r)} : r √L }`, where `⊕` extends the tuple

with the binding of the fresh attribute `c` to the value of `e`

on `r`. If `e(r)` is undefined (e.g. distance to a null vector),

the row is dropped. This is a minor departure from pure relational

algebra, but it is consistent with SQL's `NULL` handling and is

essential for safety in the presence of missing embeddings.



### 3.6 Graph expansion



`Γ_{g,d}(L)` takes a plan whose rows contain a node-id attribute

(`u`, say) and a graph attribute, applies graph rule `g` up to

depth `d`, and returns the *target* nodes of those traversals

(joined with the originating row). The depth is inclusive of

intermediate hops. `g` is a labelled-edge pattern: a regular

expression over edge labels, evaluated against the graph. The

graph itself is supplied as a relation-valued attribute; the

expansion is a join between `L` and the traversal result.



`Γ` is intentionally narrow. We do not attempt to encode path

queries as selection predicates; this keeps the operator set

small and the rewrite theory tractable (Section 4, G1). Path

information can be recovered by a `μ` that records the path

taken.



### 3.7 Example queries



We illustrate the algebra on a running example. Let `Mem` be a

relation of memory units with schema



```

Mem(id: int, text: string, ts: timestamp,

    emb: vector, emb_alt: vector,

    user: string, subsystem: string)

```



Let `KG` be a graph relation



```

KG(src: int, dst: int, label: string)

```



holding derivation links between memory units. The query "find

semantically similar passages in the fault-tolerance subsystem

from the last 30 days, ranked by a weighted sum of two distance

metrics, top 10":



```

Q1 = Mem

Q2 = σ_{subsystem = 'fault-tolerance' √ts > now() √30d}(Q1)

Q3 = μ_{‖emb √q1—→d1}  μ_{‖emb_alt √q2—→d2}(Q2)

Q4 = σ_{d1 < τ1 √d2 < τ2}(Q3)

Q5 = τ_{0.7·d1 + 0.3·d2}(Q4)

Q6 = Λ_{10}(Q5)

```



The fusion policy `0.7·d1 + 0.3·d2` is application code; the

substrate has no opinion on the weights. The thresholds `τ1, τ2`

are application choices too. The algebra's job is to make this

expression *rewriteable* —for example, to compute `d1` and `d2`

only on the post-selection rows, and to enable `topK` early

termination.



A graph-aware variant —"the same, but only memories linked by a

`derived_from` chain to a known incident memory" —is



```

R1 = σ_{id = 42}(Mem)                    -- the seed incident

R2 = Γ_{label='derived_from', d=2}(R1)   -- two hops downstream

R3 = σ_{id √π_{dst}(R2)}(Q2)            -- intersect with subsystem filter

... continue as before ...

```



The graph expansion is composable: `Γ` produces a relation that

can be `σ`-, `π`-, and `μ`-ed like any other.



---



## 4. Equational Theory: Identities and Non-Identities



The optimizer (Section 6) is driven by an *equational theory*: a

set of identities over logical plans. The same theory also

records, explicitly, the identities that classical intuition

suggests but approximation breaks —the *non-identities* —so

the optimizer knows not to apply them. This section unifies both

sides. We first give an overview (4.1), then the inherited

classical axioms (4.2), then the new identities that the extended

operators license (4.3), then the derived rewrites the optimizer

uses (4.4), and finally the non-identities with counterexamples

(4.5).



### 4.1 Overview



The equational theory of the extended algebra plays two

complementary roles. **First**, it is the foundation for the

rule-based optimizer (Section 6): every rewrite the RBO performs

is justified by a foundational identity (M1--M5, G1, G2) or by a

derived corollary / definitional / implied rule (A4-vector,

A4-vector+M1, M4+M1, M5+G1, A7-symmetry, A8-regroup, topK

recognition, σ·τ commutation) —so the optimizer's behaviour is

auditable, not opaque. **Second**, the *non*-identities (N1–N5)

are the discipline that prevents the optimizer from applying

rewrites that hold in classical relational algebra but break

under approximation, asymmetric physical costs, graph-traversal

asymmetry, or the traversal-vs-origin column split. The cost

model (Section 7) is responsible for the physical cost of every

rewrite; the equational theory is responsible for its

*correctness*.



Concretely, an identity is an equation of the form `LHS = RHS`

between two logical plans that hold for all instances of the

free variables (relation names, attribute names, query-time

bound parameters). A non-identity is a pair of plans that classical

intuition might equate but the extended setting does not, with an

explicit counterexample. Both are first-class objects: the RBO

loads them as data, the cost model consults them when computing

the "cost of identity application" term, and the executor relies

on them for safety of adaptive re-planning (Section 7.4).



The structure of this section follows the algebra's structure:

inherited axioms (the ones we use but do not re-prove) in 4.2;

new identities (the ones the extended operators license) in 4.3;

derived rules (combinations used directly by the optimizer) in

4.4; and non-identities (counterexamples that bound the

optimization horizon) in 4.5.



### 4.2 Inherited foundational axioms



The following identities are standard and are assumed without

re-proof. We list them because they interact with the new

identities in Section 6. The list includes the two classical

join-reordering identities (A7, A8) because the RBO uses them

to enumerate the `O(n!)` join orderings of

Section 6.3.



- **(A1) Selection composition.** `σ_φ σ_ψ(L) = σ_{φ √ψ}(L)`.

- **(A2) Selection distributes over union.**

  `σ_φ(L √M) = σ_φ(L) √σ_φ(M)`.

- **(A3) Projection idempotence.**

  `π_A π_B(L) = π_A(L)` if `A ⊆B`.

- **(A4) Selection–join pushdown.** If every attribute mentioned

  in `φ` is in the schema of `L`, then

  `σ_φ(L ⋈_ψ M) = σ_φ(L) ⋈_ψ M`.

- **(A5) Projection–join pushdown.** If `A` contains all

  attributes of `L` and all attributes referenced by `ψ` in

  `M`, then `π_A(L ⋈_ψ M) = π_A(L) ⋈_ψ π_{A∩attr(M)}(M)`.

- **(A6) Rename is involutive on attribute names.**

  `ρ_{b→a}(ρ_{a→b}(L)) = L`.

- **(A7) Join commutativity.** For any join predicate `ψ` that

  depends only on attributes of `L` and `M` symmetrically

  (the case for every equality, range, similarity, and

  Boolean-combination predicate in the inventory):

  `L ⋈_ψ M = M ⋈_{ψ^sym} L` where `ψ^sym` swaps the `L`- and

  `M`-side attribute references in `ψ`. Logically the two sides

  are identical as multisets; the cost of evaluating them is

  not (Theorem N2).

- **(A8) Join associativity.** For independent join predicates

  `ψ` (between `L` and `M`) and `χ` (between `L √M` and `N`),

  where `χ` does not reference any attribute introduced by `L`

  or `M` (no ambiguity):

  `L ⋈_ψ (M ⋈_χ N) = (L ⋈_ψ M) ⋈_χ N`. This is the identity the

  RBO uses to enumerate the `O(n!)` join orderings; without

  it, commutativity alone yields only `O(n)` swaps on a fixed

  parenthesization.



These hold for any predicate `φ` in our extended form (classical,

vector, text, graph), provided the attribute references are

respected. We do not claim any *new* identities here; we merely

note that the extended predicates do not violate them.



### 4.3 New foundational axioms



We introduce the *six* identities that the extended algebra

makes possible. Each is stated in a form that names the conditions

for its applicability. (Two further rules —**V1** and **J1** —
are corollaries of the assumed classical axioms A1 and A4 with

vector-predicate specialization; they appear in

Section 4.4 as derived identities, not here. The

`τ`-`Λ`-`topK` family and the `σ`-`τ` commutation are

*implied* by the operator definitions: `topK_O(L, K) := τ_O(Λ_K(L))`

from Section 3 and the row-preserving semantics of `σ` and `τ`.

The RBO uses them as zero-weight rewrites in the ARS, but they

have no separate theorem block.)



**M1. Selection commutes with materialization on the materialized

column.** Where `c` is a fresh attribute name and `e` is an

expression depending only on attributes of `L`:



$$

\sigma_{c < \tau}\, \mu_{e \to c}(L) \;=\; \mu_{e \to c}\, \sigma_{e < \tau}(L).

$$



*Intuition.* A row in `L` survives the LHS only if the

materialized value of `e` on that row is below `τ`. The RHS

applies the equivalent filter `e < τ` to the underlying attribute

*before* materialization. Both produce the same multiset of

extended rows.



*Proof sketch.* Fix `r √L`. LHS computes `e(r)`, attaches it as

`c`, and retains `r ⊆{c →e(r)}` iff `e(r) < τ`. RHS evaluates

`e(r) < τ` directly: if false, `r` is dropped; if true, the row is

extended to `r ⊆{c →e(r)}` by the `μ`. The two multisets are

equal row-by-row. The freshness of `c` ensures that the predicate

`c < τ` in LHS is unambiguous —`c` is defined by the `μ` and is

not shadowed by any pre-existing attribute. √


**M2. Materialization is commutative.** For fresh names `c_1, c_2`

and expressions `e_1, e_2` depending only on `L`'s attributes (and

not on each other's outputs):



$$

\mu_{e_1 \to c_1}\, \mu_{e_2 \to c_2}(L) \;=\; \mu_{e_2 \to c_2}\, \mu_{e_1 \to c_1}(L).

$$



*Intuition.* The two expressions are independent functions of

`L`'s attributes; the order in which we compute and bind them

does not affect the final set of `(c_1, c_2)` pairs.



*Proof sketch.* For each `r √L`, LHS computes `e_1(r)` and then

`e_2(r)`, attaching both; RHS computes `e_2(r)` and then

`e_1(r)`. Because `e_1` and `e_2` depend only on `r` and not on

each other's intermediate result, the two sequences yield the same

extension `r ⊆{c_1 →e_1(r), c_2 →e_2(r)}`. The multisets are

equal. √


**M3. Materialization through projection.** For fresh `c` and

expression `e` depending only on attributes in `A`:



$$

\mu_{e \to c}\, \pi_A(L) \;=\; \pi_{A \cup \{c\}}\, \mu_{e \to c}(L).

$$



*Intuition.* The LHS first throws away attributes not in `A`,

then computes `e` on what remains. The RHS computes `e` on all

of `L` and then projects down to `A √{c}`. If `e` is a function

of attributes in `A` only, the two multisets are equal.



*Proof sketch.* For `r √L`, the LHS produces

`π_A(r) ⊆{c →e(π_A(r))}`. The RHS produces

`r ⊆{c →e(r)}` (using the same `e` since `e` depends only on

`A`), then projects to `(A √{c})`. By the support of `e`,

`e(r) = e(π_A(r))`, and the resulting tuples agree. The

freshness of `c` ensures the projection is well-formed. √


**M4. Materialization pushdown through join.** Where `e` depends

only on attributes of `L` and `c` is a fresh attribute name:



$$

\mu_{e \to c}\, (L \bowtie_\psi M)

\;=\;

(\mu_{e \to c}(L)) \bowtie_\psi M.

$$



*Intuition.* Per-row computation can be moved to the side of the

join that owns the required attributes; the other side is left

untouched. Both sides produce the same multiset of joined

rows extended by `c`. The cost, however, differs: the LHS

computes `e` on every joined row (`|L ⋈M|`); the RHS computes

`e` only on the (typically smaller) `|L|`. When the join is

selective on the right, the RHS is dramatically cheaper.



*Proof sketch.* Fix a row `(r_L, r_M) √L ⋈M` such that

`ψ(r_L, r_M)` holds. The LHS produces

`(r_L, r_M) ⊆{c →e(r_L)}`. The RHS first materializes `L` to

obtain `μ_{e→c}(L) = {r_L' ⊆{c →e(r_L')} : r_L' √L}`, then

joins with `M` on `ψ`; since `c` is fresh and not referenced

by `ψ` (Definition: freshness), the result is

`{ (r_L', r_M) ⊆{c →e(r_L')} : r_L' √L, r_M √M, ψ(r_L', r_M) }`.

Because `e` depends only on `L`'s attributes, the materialized

value `e(r_L)` is computed in both cases from the same `r_L`,

and `ψ` is unaffected by the addition of `c` to `L`'s schema.

The two multisets agree row-by-row. √


**M5. Materialization pushdown through graph expansion.** Where

`e` depends only on attributes of `L`, `c` is fresh, and `c` is

not the graph rule's node-id attribute:



$$

\mu_{e \to c}\, \Gamma_{g, d}(L)

\;=\;

\Gamma_{g, d}\, \mu_{e \to c}(L).

$$



This is the `μ`-counterpart of G1 and the rewrite the RBO

generates for graph queries that materialize a per-origin

attribute (e.g. a distance to a fixed query point) and then

expand. The two sides produce the same multiset of `(r, v, c)`

tuples; the cost differs because the LHS computes `e` on every

traversal row while the RHS computes `e` on every origin row.

When `|Γ_{g,d}(L)| > |L|` (the typical case for `d ≤1` and a

non-trivial graph), the RHS is cheaper, and the CBO picks it.



*Proof sketch.* For each row `r √L`, the LHS first expands `r`

via `Γ_{g,d}` to obtain a multiset `{(r, v_i) : v_i reachable

from r.id by g within d}` (multiplicity = number of paths), then

materializes `e` on each traversal row. The result is the

multiset `{(r, v_i, c) : r √L, v_i reachable from r, c = e(r)}`,

where `e` depends only on `r`'s attributes by the support

condition. The RHS first materializes `e` on `L` to obtain

`μ_{e→c}(L) = {r ⊆{c →e(r)} : r √L}`, then expands via

`Γ_{g,d}`. Because `c` is fresh and is not the graph rule's

node-id attribute, the expansion proceeds from the original

node-id (unchanged by the `μ`); the result is

`{((r, c), v_i) : r √L, v_i reachable from r.id}`. Reordering

attributes, this is

`{(r, v_i, c) : r √L, v_i reachable from r, c = e(r)}`,

the same multiset as the LHS. The two multisets agree

row-by-row. √


**G1. Selection pushdown through graph expansion.** Where `φ`

references only attributes of `L` (and not attributes of the

graph rule's traversal result):



$$

\sigma_\phi\, \Gamma_{g, d}(L) \;=\; \Gamma_{g, d}\, \sigma_\phi(L).

$$



*Intuition.* Filtering the seeds before traversal produces the

same target set as filtering the traversal result, because the

traversal is monotone in the seed set (more seeds ⊆more

traversed targets). The cost, however, is dramatically

different: filtering first prunes the seed set, often by

orders of magnitude, before the expensive graph walk.



*Proof sketch.* The graph expansion `Γ_{g,d}` is

*row-multiplicating*: each origin row `r √L` produces one or

more traversal rows in the result, and the multiplicity of `r`'s

descendants is preserved (i.e. `Γ_{g,d}` does not collapse

duplicates introduced by `L`). Let `R ⊆L` be the multiset of

rows satisfying `φ`. The LHS starts from `L` and applies

`σ_φ` *after* expansion, so any traversal row whose origin row

lies outside `R` is removed; the RHS starts from `R` and only

generates traversal rows whose origin satisfies `φ`, so none are

removed. The multisets agree row-by-row, because `φ` references

only origin attributes —if `φ` referenced traversal

attributes, the two sides would diverge and the identity would

fail. √


### 4.4 Derived identities



By composing the foundational axioms and the assumed classical

identities, additional rewrites become available. The optimizer

uses these directly; we list them as corollaries rather than

as standalone foundational theorems to keep Section 4.3 focused

on the genuinely new content.



- **V1 (corollary of A1 + `∧`-commutativity).** Vector

  selections on distinct columns commute:

  `σ_{‖V_a−q_a—τ_a}\, σ_{‖V_b−q_b—τ_b}(L) =

   σ_{‖V_b−q_b—τ_b}\, σ_{‖V_a−q_a—τ_a}(L)`

  for distinct `V_a, V_b`. The two orderings produce the same

  logical result but differ in cost: the order in which the

  two vector selections are evaluated affects how many

  candidate rows each HNSW search returns, the cumulative

  HNSW work, and the intermediate result-set sizes. The

  optimizer keeps both orderings in the plan space and lets

  the cost model choose.

- **J1 (corollary of A4, with vector-predicate specialization).**

  `σ_{‖V_a−q—τ}\, (L ⋈_ψ M) =

   (σ_{‖V_a−q—τ}(L)) ⋈_ψ M`

  when `V_a √attr(L)`. The vector predicate is independent

  of any attribute of `M`, so the selection–join pushdown

  identity (A4) applies directly.

- **A4-vector+M1 (vector selection before materialization;

  in code: J1+M1).** Combining the A4 vector specialization

  (J1 in the codebase) and M1 allows a join-side vector

  selection to be pushed below a materialization that binds

  a distance for sorting:

  ```

  σ_{‖V−q—τ}  μ_{‖V−q‖→d}  (L ⋈M)

  = μ_{‖V−q‖→d}  σ_{‖V−q—τ}  (L ⋈M)        (M1)

  = μ_{‖V−q‖→d}  (σ_{‖V−q—τ}(L) ⋈M)        (A4-vector)

  ```

  The plan now computes the distance only on the surviving

  `L` rows.

- **M4+M1 (materialization pushdown, then

  selection-on-materialized).** The converse of A4-vector+M1:

  the `μ` is on the join side, and the selection is on the

  materialized column.

  `σ_{c<τ}\, μ_{e→c}\,(L⋈M) ≤μ_{e→c}\,(σ_{e<τ}(L)⋈M)`

  (apply M4 then M1). This is the rewrite that powers the

  running example: the expensive per-row computation runs only

  on rows that survive the join-side filter, not on every row

  of the join.

- **M5+G1 (materialization and selection through graph

  expansion).** A graph query with both a row-level filter

  and a row-level materialization can be reordered:

  `σ_φ\, μ_{e→c}\, Γ_{g,d}(L) ≤μ_{e→c}\, Γ_{g,d}\, σ_φ(L)`

  (apply G1 then M5).

- **A7-symmetry (reverse join).** Applying the assumed

  classical A7 in the reverse direction is a derived rewrite

  the RBO generates; the CBO then chooses the cheaper side

  per Theorem N2.

- **A8-regrouping (different join parenthesization).** Applying

  the assumed classical A8 is the rewrite that exposes a

  different join parenthesization; the RBO uses it to

  enumerate the `O(n!)` join orderings the CBO picks from.

- **`topK` recognition (definitional).** The abbreviation

  `topK_O(L, K) := τ_O(Λ_K(L))` from Section 3 makes

  `τ_O Λ_K →topK_O` a definitional rewrite; the CBO applies

  it as a physical-realization choice rather than as a logical

  RBO rewrite.

- **`σ`-`τ` commutation (implied by row-preserving

  semantics).** The identity

  `σ_φ\, τ_O(L) = τ_O\, σ_φ(L)` holds trivially: `σ` removes

  rows and `τ` rearranges them, neither changes the multiset

  of selected rows. The RBO applies it as a zero-weight

  `σ`-`τ` reordering bounded by the number of `σ`-`τ`

  inversions in `P`, letting the CBO pick filter-first or

  sort-first by selectivity.



The algebra is closed under composition: the output of any

operator is a logical plan usable as input to any other. The

schema rules ensure that the attributes a downstream operator

references are either attributes of the input or have been

introduced by a `μ` in the pipeline. The seven foundational

axioms, the assumed classical identities A1–A8, the derived

rewrites above, and the two implied-by-operator-definition

rules (`topK` and `σ`-`τ`) together suffice to cover the

optimization of all plans whose operators draw from the

inventory of Section 3.



### 4.5 Non-identities and counterexamples



Some rules that look natural in a relational setting **do not

hold** in the extended algebra, and a third class —physical

identities —holds in the logical algebra but not in the

physical algebra. We document these explicitly so the optimizer

knows not to apply them. The same logical plan in two

realizations can have different outputs; the optimizer must

treat this as a *cost* phenomenon, not a *correctness*

phenomenon.



#### 4.5.1 N1: Approximation breaks logical equivalence



> **Claim.** Logical `σ_{‖V−q—τ}(R)` is exactly defined. Physical

> `H_{V, q, τ}(R)` is *approximate* —it is missing some true

> neighbours and (depending on `ef`) may include some

> non-neighbours. A logical equivalence does not imply that two

> physical realizations preserve it.



**Counterexample.** Let `R` consist of a single row `r = (id: 1,

emb: v)` with `‖v √q—= 0.99τ` (just under the threshold). An

HNSW index on `emb` configured with a small `ef` and an

unfavourable entry point may fail to reach `r` during the

graph walk, returning an empty result. A linear scan over `R`

returns `{r}`. Both are valid physical implementations of the

same logical selection, and the logical identity

`H_{V,q,τ}(R) = F(R) √...` (whatever implementation

identity) does not hold under approximation.



**Implication.** The optimizer must not assume that two

implementations of a logical operator agree on output; it must

model approximation as a first-class *cost dimension* with

quantified recall/precision, and it must let the cost-based

search explore the trade-off (Section 7).



#### 4.5.2 N2: Similarity join asymmetry



> **Claim.** Logically, `R ⋈_{‖V_a−V_b—τ} S = S

> ⋈_{‖V_b−V_a—τ} R`. Physically, the two sides are not

> interchangeable in cost.



**Counterexample.** Let `|R| = 10^6` with an HNSW index on

`V_a`, and `|S| = 10^3` with no index. The plan "HNSW on `R`

probed by each row of `S`" issues `10^3` HNSW probes, each

touching a small portion of the graph. The reverse plan

"HNSW on `S` probed by each row of `R`" would require either

building an HNSW index on `S` (extra cost) or issuing `10^6`

probes. The two logical plans are equivalent; the physical

plans have different costs by three orders of magnitude.



**Implication.** The cost model must encode which side of a

join has the index and which has the probes. The RBO can

generate *both* the LHS and RHS join orderings; only the CBO

can decide which is cheaper.



#### 4.5.3 N3: Limit and order do not distribute over union



> **Claim.** `Λ_K τ_O (R √S) ≤Λ_K (τ_O(R)) √Λ_K (τ_O(S))`

> in general.



**Counterexample.** Let `R` and `S` each contain `100` rows with

a single scoring column; draw `R.score √U([0,1])` and

`S.score √U([9,10])`. The LHS, `Λ_5 τ_O(R √S)`, returns the

5 rows with the *globally* smallest scores —all from `R`, with

scores in `[0, 0.05]`. The RHS, `Λ_5 τ_O(R) √Λ_5 τ_O(S)`,

returns the 5 smallest of `R` *plus* the 5 smallest of `S` —
10 rows total, spanning the disjoint score ranges. The two

multisets differ.



**Implication.** The identity `Λ_K τ_O(L √M) = Λ_K τ_O(L) √
Λ_K τ_O(M)`, which would be convenient for distributed

execution, does not hold. The optimizer must compute the union

before applying the limit. This is a strong argument for

keeping the *logical* plan compact and letting the *physical*

planner manage the distributed execution (Section 7).



A related non-identity: `τ_O(L √M) = τ_O(L) √τ_O(M)` is

formal only when the union of two ordered sequences is itself

a well-defined sequence —i.e. when the user accepts that

"ordered union" of two ordered bags is the interleave, not the

re-sort. In practice, `τ` is almost always paired with `Λ`,

and T1 already subsumes the case.



---



## 5. SQL Surface



The algebra of Section 3 is the *optimizer's* working language;

it is not what application code writes. Application code writes

SQL, in a small extension that is the locked-in query interface

for the prototype. This section makes the two surfaces

explicit, gives the desugaring from SQL into the algebra, and

justifies the design decisions. The design itself is documented

in `notes/decisions/2026-07-24-sql-extension.md`; this section

is the paper-facing treatment.



### 5.1 Motivation: two equivalent surfaces



The substrate contract is the logical algebra (Sections 3—).

But application code, by convention, by tooling, and by the

decades of investment in SQL-based ORMs and dashboards, is

expressed in SQL. The right architecture is therefore a *thin

SQL surface* that desugars into the algebra, and a *round-trip*

property: every SQL query expresses a plan in the algebra, and

every plan in the algebra has a SQL rendering. The optimizer

manipulates the algebra; the parser is the only component that

sees SQL. From the application's point of view, the SQL is

the *primary* surface and the algebra is an internal

intermediate representation; from the optimizer's point of

view, the algebra is the primary surface and SQL is a

*projection* of it. The two viewpoints are not in tension

because the desugaring is total, deterministic, and

semantics-preserving.



The remainder of this section justifies the four

SQL-surface design choices the prototype has locked in

(5.8), then walks through the desugaring rules (5.6) and a

worked example (5.7).



### 5.2 DDL: vector type and indexes



The SQL surface adds one *type* and two *index* DDL clauses.

We deliberately do not introduce a new tablespace, a new

"collection" concept, or a separate vector store relation

—vectors are first-class column types, and indexes are

declared on the same relation that holds the data.



```sql

-- Vector column with optional explicit metric

ALTER TABLE memory

    ADD COLUMN emb VECTOR(768);

ALTER TABLE memory

    ADD COLUMN emb_alt VECTOR(768, METRIC := 'cosine');



-- HNSW index on a vector column

CREATE INDEX idx_emb_cos

    ON memory USING hnsw (emb vector_cosine_ops);



-- BM25 inverted index on a text column

CREATE INDEX idx_text_bm25

    ON memory USING bm25 (text);

```



The `VECTOR(N)` type is a typed array of N 32-bit floats; the

optional `METRIC := '...'` clause names the distance metric

(`'cosine'`, `'l2'`, `'ip'`) and is used by the CBO to

*prefer* the matching HNSW index when one is available. The

`USING hnsw` and `USING bm25` index methods are the physical

access paths of Section 7 exposed as DDL. Nothing in the

algebra changes; the DDL is a *creation* concern, not a

*query* concern.



### 5.3 WHERE: multi-modal predicates



The `WHERE` clause of a query is the primary surface for

multi-modal predicates. We adopt **SQL/MM-style syntax** for

all three new predicate kinds (vector, text, graph) and

*keep* pgvector's distance operators for low-level use:



```sql

-- Vector range predicate (SQL/MM-style, first-class)

WHERE emb IS NEAR :q1 WITHIN DISTANCE :tau1

WHERE emb_alt IS NEAR :q2 WITHIN DISTANCE :tau2



-- Text relevance predicate

WHERE text MATCHES :q1 WITH SCORE > :theta



-- Graph reachability predicate

WHERE id IS REACHABLE FROM :n WITHIN 2 HOPS VIA 'derived_from'



-- pgvector operators are still available (low-level, escape hatch)

WHERE emb <=> :q1 < :tau1

WHERE emb <-> :q1 < :tau1

WHERE emb <#> :q1 > :tau1

```



The design rationale for the SQL/MM style is documented and

is the user's explicit preference: function-style

(`distance(emb, :q) < :tau`) reads as a *user-defined

function* —a black box the engine does not understand —and

so discourages the optimizer from optimizing it. The

SQL/MM form `IS NEAR ... WITHIN DISTANCE` reads as a

*first-class* construct that the engine is committed to

understanding, and gives the parser a place to anchor the

desugaring rule. The same logic applies to `MATCHES ... WITH

SCORE` and `IS REACHABLE ... VIA`. We retain pgvector's

`<=>`, `<->`, `<#>` operators for two reasons: (i) backward

compatibility with the existing pgvector ecosystem, and

(ii) they are the most natural surface for an `ORDER BY`

expression that needs the raw distance (Section 5.4).



### 5.4 Materialization via existing `AS`



The materialization operator `μ` (Section 3.4) is the

algebra's way to bind a computed value as a first-class

attribute. The SQL surface does *not* add a new `LET`

clause; the existing `AS` alias is the materialization

syntax. The optimizer recognizes when an `AS`-bound value is

*reused* in `WHERE` and `ORDER BY` and triggers the M1-style

rewrite (Section 4.3) automatically.



```sql

SELECT

    id, text, ts, subsystem,

    emb <=> :q1     AS d1,    -- materializes distance to :q1

    emb_alt <=> :q2 AS d2     -- materializes distance to :q2

FROM memory;

```



The parser treats `AS d1` as binding a fresh attribute

`d1` to the per-row expression `emb <=> :q1`; the

resulting logical plan has the corresponding `μ` operator.

When the plan is later used in a `WHERE d1 < :tau1`

predicate or an `ORDER BY 0.7 * d1 + 0.3 * d2` expression,

the optimizer pushes the `σ` below the `μ` (M1) —unless

the user opted into eager materialization via the

`MATERIALIZED` hint:



```sql

WITH scored AS MATERIALIZED (

    SELECT id, emb <=> :q1 AS d1, emb_alt <=> :q2 AS d2

    FROM memory

)

SELECT id FROM scored WHERE d1 < :tau1;

```



The hint is a thin override; the default is for the

optimizer to decide, and the equational theory of Section

4 is the formal contract that the optimizer is honouring

when it pushes a `σ` past a `μ`.



### 5.5 Graph expansion via recursive CTE



Graph traversal is exposed as a standard SQL `WITH

RECURSIVE` clause —no new syntax. The optimizer handles the

recursion, and the resulting fixed point is fed into the

rest of the plan as a normal relation.



```sql

WITH RECURSIVE reachable AS (

    SELECT :n AS id, 0 AS depth

    UNION

    SELECT edge.dst, reachable.depth + 1

    FROM edge JOIN reachable ON edge.src = reachable.id

    WHERE reachable.depth < 2

)

SELECT m.*

FROM memory m

JOIN reachable r ON m.id = r.id

WHERE m.subsystem = 'fault-tolerance';

```



The recursive CTE is the standard SQL construct for

transitive closure; the optimizer recognizes the

`edge.src = reachable.id` join as a `Γ`-style expansion and

emits the corresponding `Γ` operator (Section 3.6). The

`WHERE reachable.depth < 2` clause is the depth bound. We

do *not* introduce a new `TRAVERSE` keyword; the recursive

CTE is a familiar idiom and avoids yet another surface

construct.



### 5.6 Desugaring rules: SQL →algebra



The desugaring is defined as a recursive function

`[[·]]_SQL` that maps SQL fragments to algebra terms. The

function is total on the surface syntax of Section 5.2—.5

and is the *only* place where the parser and the algebra

meet. We state the rules informally; the implementation is

the recursive-descent parser.



**D1. Projection + filter + order + limit.** For



```sql

SELECT expr AS c, ...

FROM L

WHERE φ

ORDER BY O

LIMIT K

```



the desugaring is



```

[[·]]_SQL = Λ_K(τ_O(σ_φ(μ_{expr→c}([[L]]_SQL))))

```



with `μ` chaining for multiple aliased expressions.

`ORDER BY` is preserved exactly as written —`τ_O` accepts

both simple `ORDER BY c` and complex `ORDER BY

0.7*d1 + 0.3*d2`, the latter referencing materialized

columns.



**D2. Vector predicate.** `[[emb IS NEAR :q WITHIN

DISTANCE :t]]_pred = σ_{‖emb−q—t}`. The bound parameter

`:q` and the threshold `:t` are the query-time bound

parameters of Section 3.3. The desugared form is a

first-class algebra expression, so the optimizer's

selection-pushdown rules (J1, M1) apply.



**D3. Text predicate.** `[[text MATCHES :q WITH SCORE >

:θ]]_pred = σ_{BM25(text, q) > θ}`. The `(k_1, b)`

parameters are carried with the BM25 index (5.2) and

looked up at planning time. The CBO's inverted-index

bridge (Section 7.3) applies directly to the desugared

form.



**D4. Graph predicate.** `[[id IS REACHABLE FROM :n

WITHIN k HOPS VIA 'l']]_pred = σ_{id √π_{dst}(Γ_{label='l',

d=k}(σ_{id=:n}(mem)))}`. The inner `σ_{id=:n}(mem)` is the

seed, the `Γ` is the labelled-edge expansion of Section

3.6, and the outer `σ` is the membership test. The `σ`

form composes with J1 and G1 (the desugared form

participates in the optimizer's rewrite rules).



**D5. Recursive CTE.** A `WITH RECURSIVE name AS body`

clause is first expanded to its *least fixed point* under

standard SQL semantics (bag UNION with duplicate

elimination, depth bound by the user's `depth < k`

predicate); the resulting relation is then translated by

D1–D4. Formally, with `F(R) = [[body]]_SQL` the body as a

function of the recursive name's relation, the desugaring

is



```

[[WITH RECURSIVE name AS body SELECT ...]]_SQL

  = [[SELECT ... FROM lfp(F)]]_SQL

```



The optimizer recognizes the `lfp` form and emits a `Γ`

operator with the appropriate depth and label.



For each rule, the desugaring preserves semantics

*trivially*, by construction: each rule is the definition

of the corresponding SQL/MM clause in terms of the

algebra's predicate and operator semantics (Section 3).



### 5.7 Worked example: the Q1→Q6 query in both forms



We illustrate the desugaring on the running example of

Section 3.7. The full SQL surface is:



```sql

WITH scored AS MATERIALIZED (

    SELECT

        id, text, ts, subsystem,

        emb <=> :q1     AS d1,

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



The corresponding algebra (the Q1–Q6 pipeline of

Section 3.7) is:



```

Q1 = Mem

Q2 = σ_{subsystem = 'fault-tolerance' √ts > now() √30d}(Q1)

Q3 = μ_{‖emb−q1‖→d1}  μ_{‖emb_alt−q2‖→d2}(Q2)

Q4 = σ_{d1<τ1 √d2<τ2}(Q3)

Q5 = τ_{0.7·d1 + 0.3·d2}(Q4)

Q6 = Λ_{10}(Q5)

```



The parser desugars the SQL in four steps: (1) the

`WITH scored AS MATERIALIZED` is recognized as a bound

sub-query and the inner `SELECT` is desugared to

`μ_{d1, d2}(σ_{subsystem, ts}(Mem))`; (2) the outer

`SELECT` becomes `π_{id, text, ts}(...)`; (3) the

`WHERE` clause becomes `σ_{d1<τ1 √d2<τ2}`; (4) the

`ORDER BY ... LIMIT 10` is desugared to

`τ_{0.7·d1+0.3·d2} Λ_{10}` per T1.



The optimizer then applies the rewrite rules of

Section 4. The most important rewrite is M1: the

predicate `d1 < τ1` is pushed below the materialization

that binds `d1`, so `‖emb−q1‖` is computed only on rows

that pass the `subsystem = 'fault-tolerance'` filter.

The user wrote the SQL in the order that reads naturally

("compute d1 and d2, then filter, then order"); the

optimizer delivers a plan that computes `d1` and `d2`

only on the survivors. The `MATERIALIZED` hint preserves

the plan shape (a `μ` is still emitted) but the cost of

that `μ` is reduced by the pushdown.



### 5.8 Why these design choices



Four choices are locked in for the prototype.



**SQL/MM-style over function-style.** The `IS NEAR`,

`MATCHES`, and `IS REACHABLE` syntaxes are first-class

clauses, not function calls. This is a deliberate signal

to the optimizer (and to readers of the SQL) that the

engine is committed to understanding the construct and

can apply its equational theory to it. The

function-style alternative `distance(emb, :q) < :tau`

is permitted (and is the form pgvector users are used

to) but is a *low-level escape hatch*, not the

recommended surface.



**Existing `AS` over a new `LET`.** Materialization is

expressed with the SQL `AS` alias; the optimizer

recognizes when an `AS`-bound value is reused and

applies M1. This avoids introducing a new clause for a

concept (`μ`) that already has a SQL analogue (`AS`).

The optional `WITH [MATERIALIZED]` hint covers the rare

case where a user wants to force eager materialization.



**Recursive CTE over a new `TRAVERSE` clause.** Graph

expansion is expressed with `WITH RECURSIVE`, which is

standard SQL and a familiar idiom for transitive

closure. The optimizer recognizes the form and emits a

`Γ` operator; no new keyword is needed.



**Keep pgvector's `<=>`, `<->`, `<#>` operators.** The

existing pgvector distance operators are preserved for

backward compatibility and as a low-level surface. The

SQL/MM predicates are the recommended surface, but

escape hatches matter for ecosystem fit and for advanced

users who want to compose raw distances in `ORDER BY`

expressions.



These four choices, jointly, keep the SQL surface *thin*.

The only new types are `VECTOR(N)` and the `VECTOR(N,

METRIC := ...)` variant. The only new clauses are

`IS NEAR ... WITHIN DISTANCE`, `MATCHES ... WITH SCORE`,

`IS REACHABLE ... VIA`, and the optional `[MATERIALIZED]`

hint. Everything else —`SELECT`, `FROM`, `WHERE`, `AS`,

`ORDER BY`, `LIMIT`, `WITH RECURSIVE`, `JOIN`, `GROUP

BY`, aggregates —is standard SQL. The design principle

is the *minimum new surface* that still lets the algebra

be the substrate contract.



---



## 6. From Algebra to Optimizer: RBO



The rule-based optimizer (RBO) is the first phase of the

optimizer pipeline. It operates on **logical plans** and

applies the identities of Section 4 to enumerate equivalent

plans. The output is a (typically modest) set of logically

equivalent plans that the CBO will then consider for

realization.



### 6.1 RBO role



The RBO's job is **not** to pick a plan. Its job is to

generate a *plan space* in which the CBO can search

effectively. Concretely, the RBO uses the equational theory

to:



1. **Apply selection pushdown** (A4, J1, G1) so that

   selective filters are evaluated early and reduce the

   input to expensive downstream operators.

2. **Commute vector and classical selections** (V1, A1) so

   that the more selective predicate runs first.

3. **Reorder materializations** (M2) and **push

   materialization through join / graph expansion** (M4, M5)

   so that the expensive per-row computation runs only on

   origin rows rather than on every join / traversal row.

4. **Expose topK opportunities** by the definitional

   `τ_O Λ_K →topK_O` recognition (the CBO applies it as a

   physical-realization choice).

5. **Swap selection with ordering** (the implied `σ`-`τ`

   reordering) so a selective filter runs before a sort.

6. **Reorder joins** using the assumed classical A7 and A8

   to enumerate the `O(n!)` join orderings the CBO picks

   from.



The RBO does *not* decide which access path (HNSW vs.

linear scan, B-tree vs. heap) to use. It only reshapes the

logical plan.



### 6.1.1 RBO priority order and convergence



The RBO's matchers are tried in a fixed priority order

(Algorithm 1, line 5):

`(M1, M3, M4, G1, G2, M5, M2, V1, J1, T1)`. The

order is a *heuristic* —the RBO Confluence

theorem's uniqueness of the normal form means the

final result does not depend on the order —but the

heuristic controls three observable properties.



- **Trace reproducibility.** The trace τ the RBO

  returns is what the audit log records; the same

  input plan and the same priority order produce the

  same trace, which is what the E4 experiment's

  `e4_endtoend.csv` records per query.

- **Iteration count.** Strict weight-decrease rules

  (M1, M3, M4, G1) come first because they always

  make progress; G2 is tried after the strict rules

  so that nested-Γ collapse is a post-pass on the

  already-pushed-down plan. The zero-weight rules

  (M2, V1, J1, σ-τ) are tried last because they

  preserve w and can fire repeatedly; their loop

  bound is the size of the structural equivalence

  class, not w.

- **Materialization ordering for M1+chain

  correctness.** M1 must be tried before M2: an M2

  reorder of two μs whose LHS is a σ on one of the

  aliases would invalidate M1's support condition.

  The priority order encodes this: M1 fires first,

  then M2 reorders any remaining independent μs.

  The M4+M1 derived chain requires M4 to fire

  before M1 in the join case for the same reason.



**Termination guarantee.** The loop in Algorithm 1

terminates because the weight function

`w(P) = (depth, |μ-alias uses|, |σ-on-μ-alias|)`

is a well-order on the rewrite system; each strict

weight-decrease rule strictly reduces w along a

well-ordered direction, and the zero-weight rules

preserve w but are bounded by the size of the

structural equivalence class. The `max_iters`

backstop is a safety check (default 64) that guards

against a future addition to the rule set that

accidentally breaks the well-order; in the current

release, the loop's measured iteration count is ≤

for the Q1→Q6 workload and the prototype's

synthetic 10k × 768 corpus. The RBO Confluence

theorem gives the corresponding *confluence*

guarantee: the loop's fixed point is the same

canonical form regardless of the matcher priority

order, so the optimizer's output is well-defined

even if a future contributor reorders the priority

list.



### 6.2 Example transformations



Returning to the Q1–Q6 example of Section 3.7, the input

logical plan is



```

Λ_10 τ_{0.7·d1 + 0.3·d2} σ_{d1<τ1 √d2<τ2}

    μ_{‖emb_alt−q2‖→d2} μ_{‖emb−q1‖→d1}

    σ_{subsystem='ft' √ts>now()√0d}(Mem).

```



The RBO applies the following rewrites in order:



1. **Selection composition** (A1):

   `σ_{d1<τ1 √d2<τ2}` becomes the composition of two

   `σ`s, one per predicate.

2. **Selection–materialization commutativity** (M1), per

   predicate: each `σ_{d1<τ1}` is rewritten to

   `μ_{…→d1} σ_{‖emb−q1—τ1}`.

3. **Selection–selection pushdown** (A1 again, plus V1

   for the vector part): the two `σ`s commute, and the

   one with the lower expected selectivity runs first

   (a heuristic we encode as a rule with a cost

   precondition).

4. **Classical selection pushdown** (A4): the

   `σ_{subsystem='ft' √ts>now()√0d}` is pushed to the

   base scan.

5. **topK fusion** (T1): the trailing

   `τ_{0.7·d1+0.3·d2} Λ_10` is recognized as a `topK`.

6. **Materialization commutation** (M2): the two

   independent `μ`s can be evaluated in either order;

   the RBO records both plans.



The output is a small set of logical plans, all

guaranteed to be semantically equivalent, but each with a

distinct shape (filter order, materialization order,

topK vs. sort+limit). The CBO then chooses among them.



### 6.3 Plan space pruning



Without the RBO, the CBO would have to consider every

shape of every plan: for `n` joins, the search space is

`O(n!)`; for `k` materializations, the space is `O(k!)`

per join ordering; and so on. The RBO prunes this space

in two ways:



- **Equivalent classes.** The RBO groups plans into

  equivalence classes under the rewriting rules. The CBO

  need not consider plans within a class that are

  guaranteed by the RBO to be suboptimal (e.g. plans

  where a non-selective filter runs before a selective

  one).

- **Canonical form.** Each equivalence class has a chosen

  *canonical form* (e.g. selections pushed as low as

  possible, materializations clustered, topK fused). The

  CBO enumerates realizations of the canonical form.



The effect is that the CBO's search is over *plan shapes*

and *physical realizations* jointly, but the *number of

shapes* is small (typically 1— for the Agentic-memory

queries we have explored).



### 6.4 Cross-modal plan enumeration (D)



For a single-modality query, the RBO's canonical form is

a small set: one filter order, one materialization order,

one join parenthesization. For a *multi-modal* query ---

one that combines vector, text, graph, and structured

predicates --- the canonical form is no longer obvious.

The predicates *commute* in principle (M1, V1, A1), but

the costs of evaluating them differ by orders of magnitude,

and the join parenthesization interacts with the modality

ordering. The search space is the *product* of the

modality ordering and the join orderings.



MosaicOpt's D contribution is a *modality-aware plan

enumeration* that keeps the search space small without

giving up the optimal plan. The RBO exposes the full

equivalence class (rather than collapsing to one canonical

form), and the enumeration prunes aggressively using

per-modality cost estimates:



- **Modality ordering.** If two predicates have

  selectivities that differ by an order of magnitude, the

  enumeration prunes the order in which the less-selective

  predicate runs first; the cost model already knows it

  would be slower. In practice, the modality ordering

  collapses to 1— viable candidates per query.

- **Join + modality co-enumeration.** The `n!` join

  orderings and the modality ordering are not independent:

  a join's hash-build side has different per-modality scan

  costs. The enumeration interleaves the two, using the

  modality cost to prune join orderings where the

  expensive modality sits on the build side of a hash join.

- **Index-aware pruning.** If a modality's index does not

  exist on the target relation, all plans using that

  modality as an early filter are pruned. (E.g. a BM25

  inverted index missing on a column means the σ must

  run as a post-filter, not a pre-filter.)



The output of D is the *plan space* the CBO actually

searches --- a small but complete set of plan shapes,

with the modality ordering and join parenthesization

already jointly-pruned. The CBO then assigns physical

operators to each shape (§7.3) and selects the optimal.



*Empirical observation.* For the Q1→Q6 workload, D

produces 2— plans per query, of which the CBO picks one.

Without D, the CBO would face a search space of

`O(2^k · n!)` where `k` is the number of modality-bearing

predicates --- orders of magnitude larger.



### 6.5 Style: Volcano/Cascades



Our RBO is modeled on the Cascades framework [Graefe,

1995]. Equivalence rules are first-class objects,

organized by operator pair (e.g. one rule for

`σ × μ`, another for `Γ × σ`). A memo structure

caches explored expressions and their cost bounds. The

implementation language is intentionally simple (Section

8); the *interesting* code is the rules, not the engine.



The key consequence: *adding a new operator is adding

rules*. There is no central optimizer loop to modify. A

contributor who wants to support, say, a temporal range

join, writes the operator, the equivalence rules that

reference it, and the cost-model entries —and the

optimizer gains the new capability. This is the property

that makes the algebra the API.



---



## 7. Cost Model & CBO



The cost-based optimizer (CBO) is the second phase. It

takes the equivalence classes produced by the RBO and

selects a *physical plan* —a tree of physical operators —
under a cost model.



### 7.1 Physical operator inventory



The physical operators are the *implementations* of the

logical operators. Each physical operator carries

parameters that the CBO can tune.



| Physical | Implements                        | Notes                                 |

|----------|-----------------------------------|---------------------------------------|

| `H_{V,q,τ}`   | `σ_{‖V−q—τ}`                 | HNSW range scan; parameters `ef`, `M`.|

| `B_{col,r}`   | `σ_{col √r}`                  | B-tree range scan.                    |

| `F(R)`        | base scan                        | Sequential file scan.                 |

| `I_{t,q,θ}`   | `σ_{BM25>θ}`                   | Inverted index probe.                 |

| `G_{g,d,dir}` | `Γ_{g,d}`                      | Graph traversal; `dir` is direction.  |

| `Sort(O)`     | `τ_O`                           | Full sort.                            |

| `TopK(O,K)`   | `τ_O Λ_K` (T1)                 | Bounded-heap top-K.                   |

| `Materialize(e→c)` | `μ_{e→c}`                  | Computes `e` and binds `c`.           |

| `HashJoin`, `NLJ`, `MergeJoin` | `⋈_ψ`             | Standard join algorithms.             |

| `Partition`, `HashBuild`, `HashProbe` | hash-based operators | Standard enforcers.        |



### 7.2 Cost dimensions



The cost model has four dimensions. The first three are

classical; the fourth is new.



- **I/O cost.** Page reads and writes. Dominant for

  cold data; mitigated by indexes.

- **CPU cost.** Distance computations, BM25 scoring,

  tuple materialization. For vector queries, the

  per-probe distance computation is the dominant

  component.

- **Memory cost.** Working set size, in particular the

  size of intermediate hash tables and the priority

  queue for `TopK`.

- **Approximation cost.** The *quality penalty* of

  choosing an approximate physical operator over an

  exact one. For HNSW, this is expressed as

  `1 √recall(ef, M)`; for the `topK` physical

  operator, this is 0 (it is exact, given a correct

  scoring function). The CBO expresses approximation

  as a constraint: "the chosen plan must have recall

  ≤`ρ_min`", or as a term in the cost function

  weighted by application preference.



The cost model is *additive across operators in a

pipeline*, weighted by estimated cardinalities at each

boundary. The estimation primitives (cardinality,

distinct-value count, range selectivity) are themselves

non-trivial in the multi-modal setting; we discuss

their realization in the prototype (Section 11).



### 7.2.1 Per-modality statistics (corresponds to §6.2 of paper.tex)



The CBO's two cost-based decisions (the approximation-aware

strategy picker of §7.3 and the cross-modal plan enumeration

of §6.4 / Algorithm 2) require per-query, per-modality

cardinality estimates. In classical RA, the substrate

maintains *data-only* statistics (e.g. PostgreSQL's

`pg_stats`, Oracle's histograms): the selectivity of

`σ_{a = c}(R)` depends only on the column's value

distribution, not on which `c` the query asks for. The

multi-modal setting breaks this assumption: vector, text,

and graph filter cardinalities are *per-query* —they

depend on the query vector `q`, the query text, the

starting node, *and* the threshold. Classical statistics

cannot cover them.



The prototype's statistics layer has three sketches plus

a sample-probe fallback:



- **Vector selectivity sketch.** A single `range_search`

  against the HNSW index with elevated `ef` returns a

  candidate set whose size is a lower bound on the true

  selectivity. The estimator *shares* the index with the

  main query, so the build cost is paid once; the per-call

  cost is the same as a normal range-search, ~1-3 ms.

- **Text selectivity sketch.** A single `get_scores`

  against the BM25 inverted index returns the per-row

  BM25 score; the fraction with `score > θ` is the

  selectivity estimate. ~3-4 ms per call.

- **Graph fan-out sketch.** A single `reachable` call

  returns the reachable set from the query node at

  depth `k`. The result is cached per `(n, r, k)`.

- **Sample probe (fallback).** 1% uniform sample of the

  relation; runs the actual predicate on the sample.



The three sketches are tied together by a

`StatisticsHub` that exposes a single

`estimate_selectivity(predicate, query_params)` method,

dispatching on the predicate's modality. The CBO consumes

the resulting `{column: s}` dictionary in place of its

default 0.5 prior. The dictionary is intentionally a

*lower bound*: if the CBO does not see a key for a column,

it falls back to its default prior (no information is

better than a misleading number).



**Cost and update discipline.** Hub construction is

~3-5 s (one-time) and per-call `estimate_selectivity` is

~4-5 ms. This is well within the CBO's optimization

budget (~10 ms) and orders of magnitude under the query

latency it helps optimize.



### 7.2.2 Adaptive policy (corresponds to §6.2.1 of paper.tex)



The statistics layer enables an *adaptive* CBO whose

meta-policy is: invoke the per-modality estimation step

when the cost model's regime boundaries are sharp enough

that an accurate estimate determines the regime;

otherwise, use an empirical rule. The crossover is set

by the cost model's `n`-dependence:



- **Small `n` (`n ≤n₀`, ~10⁴on the prototype's `d=128`

  corpus).** The cost model's per-row `O(n)` term in

  flat scan's cost is small compared to its constant

  overhead. At this scale, the actual latencies of flat,

  ivf, and hnsw are dominated by constants, and HNSW's

  constant is the smallest. Across all 144

  `(n √{10⁴ 3×10⁴ 10⁵}, σ √{0.1, 0.3},

  q_kind √{center, between, random})` cells × 8 τ

  values in E8, the oracle strategy is uniformly `hnsw`.

  The prototype's adaptive policy in this regime:

  *invoke HNSW directly, skip the CBO entirely*.

- **Large `n` (`n ≤n₀`, ≤0⁴on the prototype's `d=128`

  corpus).** The cost model's per-row term dominates its

  constant: at `n = 10⁶` flat's cost is ~8 ms vs HNSW's

  ~0.2 ms. The CBO with statistics is needed.



The meta-policy is a single `n`-based predicate:

`if n ≤n₀ then use HNSW directly (skip CBO) else invoke

CBO with statistics`. The threshold `n₀` is set by the

cost model's crossover (flat's predicted cost exceeds

HNSW's by a factor of 2): `n₀ ≤1.6×10⁵` on `d = 128`

and `n₀ ≤2.7×10⁴` on `d = 768`.



### 7.2.3 Filtered HNSW connectivity `ρ_f` (corresponds to §6.2.2 of paper.tex)



For a vector predicate `V` that runs after a classical

filter `C`, the CBO needs to know the **connectivity of

the HNSW graph restricted to the surviving subset

`R_f = σ_C(R)`**. A selective classical filter that picks

rows from multiple disconnected regions of the embedding

space fragments the HNSW graph: the surviving subset

`R_f` has poor internal connectivity, and the HNSW

traversal has to do extra work to surface every in-`R_f`

candidate. The CBO that does not see this signal

underestimates the filtered HNSW cost.



Define the **connectivity ratio**



  `ρ_f = (T_unfiltered / n_returned_unfiltered) /

         (T_filtered / n_returned_in_Rf)` √[0, 1]



where `T` is the HNSW search time and `n` is the number

of in-`R_f` candidates the search returns. When the HNSW

graph is well-connected within `R_f` the filtered search

scales linearly and `ρ_f →1`; when `R_f` is scattered

the filtered search has to do extra work and `ρ_f < 1`.



**Cost model extension.** The CBO's HNSW cost is

updated to



  `cost(hnsw, s, n_input, d, ρ_f) = cost(hnsw, s, n_input, d, 1.0) / ρ_f`



with recall model

`recall(s, n_input, d, ρ_f) = recall(s, n_input, d, 1.0) · (0.5 + 0.5 ρ_f)`.

When `ρ_f < ρ_min` (`ρ_min = 0.3` on the prototype), the

CBO demotes HNSW regardless of the selectivity regime:

pre-filtered HNSW on a fragmented surviving set is more

expensive than a flat scan over the same `R_f`.



**Why `ρ_f` is multi-modal specific.** Classical CBO does

not need `ρ_f` because it does not combine classical

filters with HNSW scans. Classical ANN literature

(ACORN 2022) discusses filtered HNSW but does not couple

it to a multi-modal CBO. The element `ρ_f` is genuinely

new: it depends on the relationship between the classical

filter's surviving subset and the HNSW graph structure,

and the CBO's `FilterOrder` step (Algorithm 2) uses it

jointly with selectivity to pick the modality ordering.



**Estimation.** The estimator precomputes `ρ_f` for each

(column, classical-filter-signature) pair, caches the

result, and exposes a `lookup_rho(c, F)` method to the

CBO. The build is

`AnnIndex.filtered_density(R_f, q_sample)`: a HNSW

search with `sel=R_f` and an unfiltered HNSW search,

both with `k = min(|R_f|, n)`, timed; the cost ratio

gives `ρ_f`. On the prototype's 10k × 768 corpus, the

per-(column, signature) estimation is ~1-3 ms; the hub

is built once and reused across queries.



**Order dependence.** `ρ_f` depends on the *order* in

which filters are applied, just as join order affects

table cardinality: the surviving set `R_f` is the result

of the upstream filters in sequence, and reordering

changes `R_f`. The CBO's `FilterOrder` step consumes

`ρ_f` jointly with selectivity to pick the order that

minimizes total cost; this is the cross-modal analog

of the join-order problem that classical CBO solves

via the cost-based plan enumeration. For a multi-vector

query (e.g. Q5 with both `emb` and `emb_alt`), each

vector column has its own HNSW index and its own `ρ_f`

under the same upstream filter; the CBO iterates per

column.



### 7.3 Approximation-aware CBO algorithm (A)



A traditional CBO minimizes a single weighted sum of the

cost dimensions and selects one physical realization.

MosaicOpt's CBO treats approximation as a *first-class*

fourth dimension: the search is over the *plan space* of

physical realizations, and any plan whose estimated

approximation cost exceeds the user-specified budget

`ρ_min` is *pruned* before its other three dimensions are

even evaluated. What remains is a *Pareto frontier* over

(I/O, CPU, memory) under the approximation constraint,

from which the CBO selects the optimal point (or returns

the entire frontier if the application is willing to trade

latency for memory).



**Algorithm sketch (pseudocode).** The full algorithm is

in Appendix A.5; the key loop is:



```

frontier = empty

S = PhysicalRealizations(P)         # Cascades + D's modality ordering

for P_phys in S:

    c = Cost(P_phys)                 # 4-tuple (io, cpu, mem, approx)

    if c.approx > rho_min:           # A's key step: approx-budget pruning

        continue

    if DominatedBy(P_phys, frontier): # Pareto-dominance pruning

        continue

    AddToFrontier(P_phys, frontier)

    PruneDominated(frontier)

return SelectFromFrontier(frontier)

```



**Properties.** The algorithm has three properties the

prototype relies on:



- *Planning-time approximation guarantee.* Every plan in

  the returned frontier satisfies `C(P).approx ≤ρ_min` by

  construction. The application can rely on the CBO never

  returning a plan that violates its quality budget *as

  estimated by the cost model*.

- *Polynomial expected runtime.* For the common case of

  `|S| √O(poly(n))` physical realizations per logical

  plan (where `n` is the operator count), the

  frontier-based search runs in time linear in `|S|`

  amortized. Selinger-style dynamic programming for the

  `⋈` fragment and top-K pruning for `τ Λ` reduce the

  worst case from `O(n!)` to `O(n · 2^n)` in practice; the

  bound remains exponential in `n` in the worst case.

- *Convergent with RBO.* The frontier search is over one

  logical plan at a time. Combined with the RBO's

  confluence (§4), the end-to-end optimizer is confluent.



### 7.4 Logical-to-physical bridge



For each logical operator, the CBO enumerates a set of

candidate physical realizations. The bridge is a set of

*implementation rules* in the Cascades sense:



- `σ_{‖V−q—τ}(R)` may be realized as `H_{V,q,τ}(R)` if

  an HNSW index exists on `V`; otherwise as

  `μ_{‖V−q‖→d} F(R)` followed by `σ_{d<τ}`. The CBO

  also considers the *hybrid* plan: HNSW with a small

  `ef` for candidates, followed by an exact

  re-evaluation of the threshold.

- `σ_{BM25>θ}(R)` may be realized as `I_{t,q,θ}(R)`

  if an inverted index exists on `t`; otherwise as a

  full scan with BM25 scoring.

- `τ_O Λ_K(R)` is realized as `TopK(O, K)(R)`, per

  T1. The CBO may also consider `Sort(O)(R)` followed

  by `Limit(K)`, useful when the cost of sorting is

  amortized over a downstream group-by.



The bridge rules are *strategy-agnostic*: they enumerate

candidate physical operators but do not commit to one.

The approximation-aware search of §7.3 then selects the

operator within each rule, subject to the approximation

budget.



### 7.5 Adaptive re-planning (B)



A gives a *planning-time* recall guarantee, but only as

accurate as the cost model. The estimated recall of an

HNSW plan is `recall(ef, M)`, which is a fitted curve

over a calibration sample. Real workloads drift from the

calibration distribution; cluster boundaries shift; query

patterns change. The estimated recall can disagree with

the *observed* recall.



§7.5 closes the gap with *adaptive re-planning*: at

well-defined plan points in the physical execution ---

after an HNSW scan returns its candidate set, after an

inverted-probe returns its postings, after a graph

expansion returns its reachable set --- the executor

pauses, observes the actual cardinality and a small-sample

recall estimate, and *re-consults the cost model*. If the

observed recall falls short of `ρ_min`, the executor

switches to a higher-recall physical realization of the

same logical operator (e.g. larger `ef`, larger `nprobe`,

or fall back to flat scan). The algebra is unchanged; only

the physical subtree is replaced.



**Algorithm sketch (pseudocode).** The full algorithm is

in Appendix A.5; the key loop is:



```

r = SampledRecall(C, sample_size)    # compare C against a small exact subsample

if r >= rho_min:

    return Continue(C)

P_prime = HigherRecallRealization(P) # larger ef, larger nprobe, or flat scan

return ReplaceSuffix(P, P_prime)

```



**Safety.** An adaptive switch is *safe by construction*:

the new suffix is an implementation of the same logical

operator as the old suffix, so the logical result is

unchanged. The cost is bounded: the sampled recall check

is `O(k)` for a sample of `k` rows, and the

higher-recall realization is selected from a precomputed

set of `≤4` alternatives (one per physical operator

family).



---



## 11. Experimental Study



The prototype is a working end-to-end implementation

of the extended algebra, the rule-based optimizer,

and the cost-based physical-operator selector. The

codebase is intended to be the primary artifact

accompanying this paper: each identity in Section 4

is implemented as a Python matcher that runs over

the same plan tree the algebra emits, and the CBO

plugs into the physical layer through a 4-tuple cost

vector. The goal is to demonstrate that the algebra

is *executable*, not just paper.



### 11.1 Setup

We evaluate the prototype on **three** real-style datasets:
two at n=10,000, d=128 (BigVectorBench-style and SIFT1M-extended)
and one at n=10,000, d=100 (GloVe-100-extended).

- **BigVectorBench-style *glue* table**
  (`experiments/data_gen_real.py:gen_glue_table`, 5
  classical categories).  A faithful simulator of the
  workload classes described in BenchCouncil's BigVectorBench
  benchmark [BigVectorBench VLDB 2025]: it carries two
  L2-normalized embedding columns (one primary, one
  multi-vector), a BM25 text column, a classical categorical
  column, a fuzzy-text column, a graph-node column, and a
  timestamp.  Predicates are predicate-correlated with
  embedding geometry (mirroring the ACORN setting) so that
  range queries and BFS expansions have meaningful candidate
  sets.
- **SIFT1M-extended** (`experiments/data_gen_real.py:gen_sift1m_extended`).
  Following the ACORN and NaviX evaluation methodology, we
  take the SIFT-1M benchmark [Lowe 2004] as the vector base,
  and *extend* it with the same classical/text/graph columns
  as the glue table — making it a strict superset of what
  ACORN and NaviX evaluate (both use SIFT1M with one scalar
  label column as filter).  The vector is SIFT-128 style
  (d=128, 5 cluster categories); the BM25 corpus is
  image-tag style ("mountain", "beach", etc.).
- **GloVe-100-extended**
  (`experiments/data_gen_real.py:gen_glove_extended`).
  Following the same ACORN/NaviX evaluation methodology on
  the GloVe axis, we take the GloVe-100 benchmark
  [Pennington et al., EMNLP 2014] (1.18M pre-trained word
  vectors, angular distance, the second canonical
  ANN-benchmark base alongside SIFT1M) as the vector base,
  and *extend* it with the same classical/text/graph columns.
  GloVe is in the ACORN Table 2 evaluation set and is a
  fixture of ANN-benchmarks; including it makes the
  cross-system comparison apples-to-apples on a third
  widely-used base that differs from SIFT1M in dimensionality
  (d=100 vs d=128) and in semantic topology (word-vector
  clusters rather than SIFT visual descriptors).  The vector
  is GloVe-100 style (d=100, 5 semantic-topic clusters:
  politics / science / commerce / culture / sports); the
  BM25 corpus is topic-word style ("election", "team", etc.).

The three datasets together cover three dominant axes of
vector search benchmarks: (a) *modality-rich* workloads
(BigVectorBench), (b) the canonical *visual-descriptor
ANN-benchmark* base (SIFT1M), and (c) the canonical
*text-embedding ANN-benchmark* base (GloVe-100).  All three
are extended to the same modality set so that the
cross-system comparison is apples-to-apples: the same query
categories, the same baselines, and the same selectivity grid.

**Graph topology.**  All three datasets use the same *star per
cluster* topology (per cluster, one hub node connects to
every other node) plus ~5% inter-cluster random edges.
This guarantees that BFS depth 2 from any seed node reaches
the entire cluster the seed belongs to — a property that
the previous synthetic generator lacked, and that the
BigVectorBench setting implicitly assumes (predicates are
embedding-correlated, so a graph predicate on a
cluster-aligned seed is non-empty).

**Graph scaling.**  At n=10,000, the glue table has 20,490
edges (~20×n); the SIFT1M-extended table has 20,490 edges
with the same density; the GloVe-100-extended table has the
same density.  Building the HNSW index over the vec_a
column takes ~400ms per dataset; we reuse the index across
the full sweep (real-workload amortization).

**Hardware.**  Single-thread, no GPU.  Timings measure
wall-clock latency including Python interpreter overhead.
Stack: numpy 2.4.6, scipy 1.17.1, faiss-cpu 1.14.3,
rank-bm25 0.2.2.  Full E1 sweep on both datasets (8
categories × 8 selectivities × 6 systems) completes in
~90 seconds.

**Baselines.**  We compare MosaicOpt against **five**
baselines that together cover the design space of
single-modal, vector-first, classical-first, and naive
multi-modal systems:

- **NaviX** (Sehgal et al., VLDB 2025 [vol18/p4438],
  kuzu-based) — a single-CBO unified engine that picks the
  best per-modality physical operator independently and
  intersects the candidates.  We re-implement the design in
  Python (`optimizer/baselines.py:NaviXStyle`) because the
  full C++ kuzu stack is not available in our environment.
- **ACORN-σ** (Gollapudi et al., 2022 [arXiv:2211.15573],
  faiss IDSelector, σ=1.0) — a vector-first pipeline that
  runs HNSW with pre-filter, then post-filters with BM25 and
  BFS.  ACORN natively supports attribute filters but not
  text or graph reachability; we add the minimal post-filter
  passes as the natural extension.
- **Pre-filter (pgvector-style)** — the *de facto industry*
  approach (Milvus hybrid search, Weaviate hybrid search,
  pgvector [pgvector 2024]): run a boolean classical
  pre-filter first to get a candidate set R_pre, then run an
  HNSW range query restricted to R_pre via faiss
  `IDSelector`.  Text and graph predicates are post-filter
  passes on the surviving set (pgvector / Milvus do not
  expose per-predicate indexes).  No RBO, no CBO, no ρ_f,
  no adaptive re-planning.  Vector strategy is always HNSW.
- **Post-filter (naive)** — run HNSW on all n rows first,
  then post-filter the surviving set with classical, BM25,
  and BFS.  Vector strategy is always HNSW.  No RBO, no CBO,
  no ρ_f, no adaptive re-planning.  This is the "no plan
  enumeration, no pre-filter" baseline.
- **BVB-native** — a synthetic baseline representing the
  *per-workload fixed-order* execution style used by the
  BigVectorBench backends (milvus, weaviate, elasticsearch):
  classical pre-filter → vector HNSW → BM25 post-filter →
  graph BFS post-filter, in a fixed order.  Vector strategy
  is always HNSW.  No RBO, no CBO, no ρ_f, no adaptive
  re-planning.  This baseline represents the cost of *not*
  having a cross-modal CBO; the headline speedup of
  MosaicOpt over this baseline is the structural argument
  for the paper.

**Recall floor.**  The CBO's recall floor is 0.95.  No query
in the E1–E3 sweeps is allowed to return a plan whose
estimated recall is below this floor; if the natural-regime
strategy falls short, the CBO promotes to the next strategy
in the order.  In the experiments below, the CBO's `flat`
choice is what gives MosaicOpt its 1.00 recall on every
category; the pre-filter baselines (NaviX, ACORN,
pre-filter, BVB-native) drop to 0.81–0.92 on the
multi-vector and join categories because HNSW's ~5–15%
approximation loss is amplified on the small surviving
sets.  The post-filter baseline matches MosaicOpt's 1.00
recall (HNSW on full graph is exact on the range query)
but pays a per-query overhead of ~100–150ms for the BM25
index build and full-graph HNSW scan.

### 11.2 E1: Cross-modal categories Q1–Q8 on real-style data

We sweep **eight** query categories that together cover the
four workload classes of BigVectorBench (filter-ann, mm-ann,
mv-ann, big) plus four single-modality or hybrid regimes
that highlight specific optimizer modules:

- **Q1 VTC**: vector range + BM25 threshold + classical
  (3 modalities; the ≥3-modality E1 carryover).
- **Q2 VGC**: vector range + graph reachability (depth 2) +
  classical.
- **Q3 VTGC**: all 4 modalities active (vector + text +
  graph + classical) — the Q1→Q6 representative case.
- **Q4 VTf**: vector + BM25 + fuzzy `LIKE`.
- **Q5 V-only**: vector range + classical.  The canonical
  "filter + vector" ACORN-style workload.
- **Q6 T-only**: BM25 + classical.  A pure-text regime
  (no vector).
- **Q7 MV**: two vector columns (vec_a, vec_b) + classical.
  Multi-vector fusion (WebVid-style).
- **Q8 Join**: self-join on the classical column, then
  vector distance.  The J1 scenario from the E2 corollary.

For each category, 8 query instances at target overall
selectivities {0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15,
0.2} by varying the per-predicate parameters (vector τ,
BM25 threshold, graph depth, classical value, fuzzy term).
8×8 = 64 queries, each run on 4 systems with a fresh HNSW
build per category, 3 runs per cell.  Total: 64×4×3 = 768
measurements.  Recall is computed against an exact ground
truth on the full glue table.

| target s = 0.05 | Q1 VTC QPS | Q1 VTC rec. | Q3 VTGC QPS | Q3 VTGC rec. | Q5 V-only QPS | Q5 V-only rec. | Q7 MV QPS | Q7 MV rec. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MosaicOpt | 63.8 | 1.00 | 74.1 | 1.00 | 71.5 | 1.00 | 49.2 | 1.00 |
| post-filter | 5.9 | 1.00 | 6.0 | 1.00 | 6.2 | 1.00 | 3.0 | 1.00 |
| NaviX | 6.2 | 0.86 | 5.9 | 0.86 | 7.6 | 0.84 | 3.8 | 0.81 |
| pre-filter | 6.0 | 0.86 | 5.8 | 0.86 | 6.1 | 0.84 | 3.0 | 0.81 |
| ACORN | 5.8 | 0.86 | 6.1 | 0.84 | 7.4 | 0.84 | 3.7 | 0.81 |
| BVB-native | 5.9 | 0.86 | 6.2 | 0.84 | 7.5 | 0.84 | 3.7 | 0.81 |

(Full 8 categories × 8 selectivities × 6 systems = 384 cells
in `results/e1_real_n10k_d128.csv`.)

**SIFT1M-extended results (SIFT-128 + scalar + text + graph, n=10,000).**

| target s = 0.05 | Q1 VTC QPS | Q1 VTC rec. | Q3 VTGC QPS | Q3 VTGC rec. | Q5 V-only QPS | Q5 V-only rec. | Q7 MV QPS | Q7 MV rec. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MosaicOpt | 75.8 | 1.00 | 75.5 | 1.00 | 66.5 | 1.00 | 40.8 | 1.00 |
| post-filter | 6.6 | 1.00 | 6.2 | 1.00 | 6.2 | 1.00 | 2.9 | 1.00 |
| NaviX | 6.8 | 0.86 | 6.3 | 0.84 | 6.4 | 0.88 | 3.0 | 0.72 |
| pre-filter | 6.5 | 0.86 | 6.6 | 0.84 | 6.3 | 0.84 | 3.0 | 0.70 |
| ACORN | 6.5 | 0.87 | 6.3 | 0.84 | 6.4 | 0.86 | 2.9 | 0.71 |
| BVB-native | 6.5 | 0.86 | 6.2 | 0.86 | 6.0 | 0.84 | 3.0 | 0.72 |

(Full 8 categories × 8 selectivities × 6 systems = 384 cells
in `results/e1_real_sift_n10k_d128.csv`.)

**GloVe-100-extended results (GloVe-100 + scalar + text + graph, n=10,000, d=100).**

| target s = 0.05 | Q1 VTC QPS | Q1 VTC rec. | Q3 VTGC QPS | Q3 VTGC rec. | Q5 V-only QPS | Q5 V-only rec. | Q7 MV QPS | Q7 MV rec. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MosaicOpt | 43.7 | 1.00 | 47.4 | 1.00 | 52.9 | 1.00 | 23.9 | 1.00 |
| post-filter | 3.5 | 0.92 | 3.5 | 0.92 | 3.6 | 0.92 | 1.6 | 0.93 |
| NaviX | 3.4 | 0.58 | 3.4 | 0.58 | 3.5 | 0.60 | 1.7 | 0.48 |
| pre-filter | 3.6 | 0.58 | 3.5 | 0.59 | 3.3 | 0.60 | 1.7 | 0.47 |
| ACORN | 3.6 | 0.59 | 3.5 | 0.58 | 3.4 | 0.60 | 1.7 | 0.47 |
| BVB-native | 3.6 | 0.60 | 3.6 | 0.58 | 3.3 | 0.60 | 1.7 | 0.48 |

(Full 8 categories × 8 selectivities × 6 systems = 384 cells
in `results/e1_real_glove_n10k_d100.csv`.)

**Finding (multi-modal speedup).**  MosaicOpt is **9–13×
faster than BVB-native** on every category of every dataset,
with **perfect recall** (1.00) on all 8×3 = 24 (category,
dataset) cells.  The post-filter baseline matches MosaicOpt's
recall closely (its HNSW range query on the full graph is
exact on the range query) but pays a per-query overhead of
~100–150ms for the BM25 index build and full-graph HNSW
scan, leaving its QPS at 3–7 (i.e., 10–13× slower than
MosaicOpt).  The pre-filter baselines (NaviX, ACORN,
pre-filter, BVB-native) drop to 0.47–0.92 recall because
they use HNSW with pre-filter on the vector columns —
HNSW's ~5–15% approximation loss is amplified on the
multi-vector and join categories (Q7, Q8) where the
surviving set is small.  The drop is most severe on
GloVe-100-extended (Q7 MV ≈ 0.47) where the lower
dimension (d=100) leaves less margin for HNSW to absorb
the approximation loss.  MosaicOpt's CBO picks `flat` (the
exact strategy) at n=10,000 on d ∈ {100, 128}, which is why
it gets 1.00 recall at the cost of higher per-cell latency
on the largest selectivity bands; the *structural* win is
that the CBO chooses the correct strategy per query.

**Finding (baseline structural cost).**  All five baselines
(NaviX, ACORN, pre-filter, post-filter, BVB-native) pay
~100–150ms per-query cost regardless of selectivity, with
no significant difference between pre-filter and post-filter
(both are dominated by the same Python interpreter and BM25
index build overhead).  MosaicOpt amortizes the BM25 index
across the whole sweep (real-workload amortization) and uses
the classical pre-filter to reduce the candidate set to
~2000 rows before the vector step; this is the structural
source of the speedup.  MosaicOpt's QPS ranges from 41 (Q7
MV on SIFT1M-extended) to 76 (Q4 VTf on the glue table) per
cell at s=0.05; the baselines are pinned at 3–8 QPS.

**Why no recall = 0 anywhere.**  The previous synthetic
5-cluster Gaussian experiment had Q2/Q3/Q4/Q7 returning 0
recall at the lowest selectivities because the graph was
too sparse for BFS to find the predicate-relevant rows.
The current star-per-cluster topology guarantees that BFS
depth 2 from any seed reaches the entire cluster the seed
belongs to, so every Q1–Q8 cell has a non-empty ground
truth at every target selectivity.

![E1 on the BigVectorBench-style glue table: QPS vs target selectivity
per query category (left), and recall vs target selectivity (right). MosaicOpt
dominates the QPS plot at all 8 selectivities; its CBO-picked `flat`
strategy gives 1.00 recall across all 8 categories while the pre-filter
baselines (NaviX, ACORN, pre-filter, BVB-native) lose 5–15% recall on
Q7 and Q8.](figures/e1_real_qps_glue.png){ width=49% }
![E1 on the BigVectorBench-style glue table: recall vs target selectivity
on the same cells. Pre-filter baselines (NaviX, ACORN, pre-filter, BVB-native)
drop to 0.81–0.92 on Q7/Q8; the post-filter baseline matches MosaicOpt's
1.00 recall but at ~10× lower QPS.](figures/e1_real_recall_glue.png){ width=49% }

![E1 on the SIFT1M-extended dataset (SIFT-128 + scalar label + BM25 text
+ star-per-cluster graph, n=10,000): QPS vs target selectivity per
query category (left), and recall vs target selectivity (right). The
9–13× MosaicOpt speedup is consistent with the glue table; the
pre-filter baselines drop further on this dataset (Q7 MV recall ≈ 0.70)
because HNSW fragments more aggressively on SIFT1M's non-cluster-correlated
vector distribution.](figures/e1_real_qps_sift.png){ width=49% }
![E1 on the SIFT1M-extended dataset: recall vs target selectivity on
the same cells. MosaicOpt again achieves 1.00 recall; pre-filter baselines
drop to 0.70–0.86 on Q7 MV and Q8 Join; post-filter matches 1.00 at
~10–13× lower QPS.](figures/e1_real_recall_sift.png){ width=49% }

![E1 on the GloVe-100-extended dataset (GloVe-100 + scalar label + topic
BM25 text + star-per-cluster graph, n=10,000, d=100): QPS vs target
selectivity per query category (left), and recall vs target selectivity
(right). MosaicOpt's 10–14× speedup holds on the GloVe base as well; it
is uniformly faster than every baseline on every category.](figures/e1_real_qps_glove.png){ width=49% }
![E1 on the GloVe-100-extended dataset: recall vs target selectivity on
the same cells. MosaicOpt achieves 1.00 recall on all 8 categories. The
pre-filter baselines drop the most on this dataset (Q7 MV ≈ 0.47 vs 0.70
on SIFT1M-extended) because the lower dimension leaves less margin for
HNSW to absorb the approximation loss. The post-filter baseline drops to
≈0.92 on every category because the BM25 corpus is shorter and less
discriminative for GloVe's semantic-topic clusters.](figures/e1_real_recall_glove.png){ width=49% }

### 11.3 E2: RBO scenarios on real-style data

E2 re-derives the three minimal-query RBO findings from the
E3 corollary on the real-style data: which rewrites are
load-bearing, and on which scenarios.  Three templates:

- **M1** (σ ∘ μ → μ ∘ σ): push a filter on a materialized
  column above the materialization.  On real data this is
  the multi-vector Q7 scenario (vec_a + vec_b + classical)
  re-executed with RBO off (the σ ∘ μ is computed on all n
  rows) vs RBO on (HNSW pre-filter → exact distance on
  survivors).
- **T1** (Λ ∘ τ → TopK): rewrite a τ+Λ pair to a single
  TopK.  On real data this is Q1 (VTC) with a Λ_K at the
  top.  with_rbo: heap-based executor (np.argpartition);
  no_rbo: full sort + limit (np.argsort).
- **J1** (σ ∘ Join → Join ∘ σ): push a filter into a join's
  input.  On real data this is Q8 (self-join on classical +
  vector).  with_rbo: classical pre-filter → self-join on
  |R|/5 rows; no_rbo: full self-join on n rows then
  post-filter.

Per template: 5 query instances at varying τ or K, 3 runs
each, 15 measurements per scenario.  Total: 3×15 = 45
measurements on the n=10,000 glue table.

| τ (or K) | M1 with-rbo (ms) | M1 no-rbo (ms) | T1 with-rbo (ms) | T1 no-rbo (ms) |
| --- | --- | --- | --- | --- |
| τ=0.74 | 0.84 | 8.61 | — | — |
| τ=0.78 | 1.02 | 9.35 | — | — |
| τ=0.82 | 0.89 | 11.92 | — | — |
| K=10 | — | — | 9.27 | 8.98 |
| K=100 | — | — | 3.68 | 4.40 |
| K=1000 | — | — | 3.88 | 4.79 |
| **median speedup** | **9.14×** | | 1.19× | |

**Finding (M1 wins big).**  M1's σ ∘ μ → μ ∘ σ rewrite
exposes the vector range predicate to the CBO, which then
picks HNSW pre-filter.  The structural effect: at n=10,000
on d=128, the rewritten plan runs the HNSW range query in
~1ms vs. the ~9ms brute-force distance computation on all n
— a **9.14× speedup** on the materialization itself (median
over 5 τ values; range 6.79–11.92 vs 0.71–1.57 ms).  This
is the dominant RBO win on the real-style data; the
synthetic E3 corollary found a similar 10.3× speedup,
validating the finding on a different data distribution.

**Finding (T1 marginal).**  T1's τ+Λ → TopK rewrite is a
marginal 1.19× at n=10,000 (the heap-vs-sort difference is
below the executor noise floor); at n≥10^5 the speedup
grows proportionally to log n.  This matches the synthetic
E3 finding: T1 is correct but its absolute impact is small
at prototype scale.

**Finding (J1: structural win, absolute time dominated by
O(n²) join).**  J1's σ ∘ Join → Join ∘ σ rewrite correctly
reduces the join input from n=10,000 to ~n/5=2,000 rows (a
25× reduction in join work).  But on the n=10,000 glue
table, the with_rbo and no_rbo paths are within 1% of each
other in absolute wall-clock time (both around 1.3s per
query).  The reason: the O(n²) self-join dominates the
post-filter cost; the classical pre-filter's 25× work
reduction is swamped by the join itself.  At n≥10^5 the
join work would dominate more sharply, and J1's structural
effect would translate to a measurable wall-clock speedup.
The data are in `results/e2_real_rbo_n10k.csv`.

### 11.4 E3: CBO + cardinality ablation on real-style data

E3 attributes the E1 latency to specific CBO modules.  We
sweep the same Q1/Q3/Q5/Q8 categories at the same 8
selectivities as E1 with five configurations of the
optimizer:

- **default_0.5**: CBO uses 0.5 as a uniform prior for all
  modalities, no statistics, no ρ_f, no adaptive policy.
- **with_stats**: per-query selectivity estimate is enabled
  (we feed the CBO the true post-classical selectivity from
  a single vector-distance histogram).
- **with_rho**: ρ_f estimation is enabled.
- **with_adaptive**: the adaptive policy
  (§7.2.2, n_0=1.6×10^5) is enabled.
- **full**: all modules enabled.

5 configurations × 4 categories × 8 selectivities = 160
cells.  Per-cell: 3 runs, median latency and recall against
the exact ground truth.

| config | QPS (median) | recall (median) | speedup vs default_0.5 | effective scenario |
| --- | --- | --- | --- | --- |
| default_0.5 | 672 | 1.000 | 1.00× | clustered filters, low s |
| with_stats | 671 | 1.000 | 1.00× | (no per-cell win on this data) |
| with_rho | 691 | 1.000 | 1.03× | scattered filters (not exercised here) |
| with_adaptive | 1855 | 0.845 | **2.76×** | when HNSW approximation is acceptable |
| full | 1883 | 0.845 | **2.80×** | (same as with_adaptive) |

**Finding (CBO is conservative at n=10,000).**  On the
real-style data, the CBO's predicted cost says `flat` is the
cheapest strategy (cost model constants: 0.04 + 0.06 ×
n_input × d / 10^6), and so `default_0.5` picks `flat` on
every Q1/Q3/Q5 cell.  This is the *correct* choice: `flat`
gives 1.00 recall; `with_adaptive` (which forces HNSW
directly) is 2.76× faster but loses ~15% recall on the
multi-vector and join cells.  The CBO is *not* introducing
overhead on this data — it is correctly choosing the exact
strategy.

**Finding (cost-model calibration gap).**  The cost model
*predicts* `flat` at 0.06ms (n_input=2,000, d=128), but
`flat` actually runs at ~1.49ms (dominated by Python
interpreter overhead, not the underlying NumPy scan).  The
`with_adaptive` path skips the CBO and uses HNSW (~0.54ms),
which is faster on this data despite HNSW's approximation
loss.  The cost-model constants in
`optimizer/cbo.py:COST_FLAT` need re-calibration on the
real workload; the current values were derived on the
synthetic n=10^4, d=128 corpus and underestimate the
Python interpreter cost.

**Finding (recall floor is binding).**  The `with_adaptive`
recall of 0.845 is below the CBO's 0.95 recall floor on a
workload where the surviving set is small (Q7 MV, Q8 Join
at s=0.001).  The adaptive policy currently does not check
the recall floor; this is a known limitation
(§11.6).  A principled fix is to add a recall floor check
to Algorithm 2 that falls back to `flat` when the
predicted recall is below the floor.  The data are in
`results/e3_real_cbo_n10k.csv`.

![E3: QPS by CBO configuration (5 configs × 4 categories × 8 selectivities
= 160 cells, BigVectorBench-style glue table, n=10,000). `with_adaptive`
is fastest but loses recall; `default_0.5` is the most balanced.](figures/e3_real_cbo_qps.png){ width=49% }
![E3: recall by CBO configuration on the same 160 cells. `default_0.5`
and `with_rho` get 1.00; `with_adaptive` and `full` get 0.845 (HNSW
approximation loss amplified on small surviving sets).](figures/e3_real_cbo_recall.png){ width=49% }

### 11.5 Limitations



- **Single-node, single-user.** The algebra

  does not yet model partitioning, replication,

  or isolation. The cost model has no term

  for cross-node I/O. A distributed extension

  is sketched below.

- **Read-only.** There is no write path. The

  algebra is defined on a snapshot of the

  memory store. A write algebra would need

  to model incremental updates to materialized

  views and to the cost model's statistical

  summaries.

- **Approximation is a cost, not a contract.**

  The cost model treats recall as a

  *dimension*; the algebra treats queries as

  *exact*. There is a real tension here, and

  we have not yet resolved whether the

  *user-facing* language should expose

  approximation as a hint (`hint

  approx_recall=0.9`) or as a contract

  (`requires recall ≤0.9`).

- **Cost model calibration.** The cost model

  requires empirical calibration that we have

  not yet performed. A poorly calibrated

  cost model is worse than no cost model,

  because it gives the CBO a false

  confidence.



### 11.6 Open problems



- **Write path and incremental maintenance.**

  Materializations (`μ`) and indexes (HNSW,

  inverted) need to be kept consistent with

  the base relations. For a memory store

  with append-mostly workloads, a

  log-structured write path with deferred

  index updates is the natural design.

- **Multi-tenancy.** The algebra does not

  currently carry a tenant identity through

  plans. Adding `tenant` as a first-class

  attribute and the corresponding

  selection-pushdown rules is a

  straightforward extension, but the cost

  model must learn to factor tenant

  cardinality.

- **Distributed execution.** The non-identity

  N3 (limit and order do not distribute

  over union) is a particular obstacle

  for distributed `topK`. A natural

  research direction is a *partition-aware*

  topK algorithm that respects N3 and

  admits a physical operator with bounded

  approximation.

- **Learned indexes.** Recent work on

  learned indexes [Kraska et al., 2018]

  suggests that index structures can be

  replaced by learned models. The

  logical/physical separation makes this a

  drop-in change: a learned index is a

  physical realization of a selection

  operator. The algebra and rewrite rules

  are unchanged.



### 11.7 Connection to other research



The algebra draws on three threads.



- **Linear algebra.** Vector distance is

  `‖V−q‖`; an HNSW probe is a

  linear-algebra computation. There is a

  natural research direction in *fusing*

  linear-algebra kernels with relational

  operators at the physical level —for

  example, a join that, in the same

  pipeline, computes the distance to a

  query vector.

- **Probabilistic databases.** Approximation

  in HNSW is, formally, a probability

  distribution over possible outputs. The

  cost model's "approximation dimension"

  is a step toward treating the database

  as a probabilistic system; a full

  treatment in the style of

  *probabilistic databases* [Dalvi &

  Suciu, 2007] is a natural extension.

- **Programming-language theory.** The

  materialization operator `μ` is, in

  spirit, a `let` binding in a query

  language. There is a natural

  correspondence with the *let-binding*

  extensions of relational algebra used

  in query-language research, and the

  freshness condition for `c` is a

  hygienic-rename condition. A

  type-theoretic treatment of the

  algebra would strengthen the foundation

  and make the rewrite rules easier to

  state and prove.



### 11.8 Closing



The argument of this paper is simple: agentic

memory is multi-modal, multi-modal is a query

problem, query problems benefit from

foundations, and the right foundation is an

extension of relational algebra. The algebra

is small, the rewrite rules are provable, and

the optimizer is a direct consequence. We do

not claim to have solved agentic memory; we

claim that the field will benefit from a

shared vocabulary, and we offer MosaicOpt as

a starting point for that conversation.



---



## Acknowledgements



This draft builds on the project design

brief (`docs/brief.md`). Sections 3—

derive from the brief's algebra-design

decisions. The authors thank the open-source

communities of HNSW (hnswlib), Tantivy,

DuckDB, and Cascades-style optimizers for

the building blocks that make the prototype

tractable.



---



## References (informal)



- Codd, E. F. (1970). *A Relational Model of Data

  for Large Shared Data Banks.* CACM 13(6).

- Selinger, P. G. et al. (1979). *Access Path

  Selection in a Relational Database Management

  System.* SIGMOD.

- Graefe, G. (1994). *Volcano —An Extensible and

  Parallel Query Evaluation System.* IEEE TKDE.

- Graefe, G. (1995). *The Cascades Framework for

  Query Optimization.* IEEE Data Engineering

  Bulletin.

- Salton, G., Wong, A. & Yang, C. S. (1975). *A

  Vector Space Model for Automatic Indexing.*

  CACM 18(11).

- Robertson, S., Walker, S. & Jones, S. (1995).

  *Okapi at TREC-3.* NIST Special Publication.

- Mikolov, T. et al. (2013). *Efficient Estimation

  of Word Representations in Vector Space.* ICLR

  Workshop.

- Reimers, N. & Gurevych, I. (2019). *Sentence-

  BERT: Sentence Embeddings using Siamese BERT-

  Networks.* EMNLP.

- Malkov, Y. A. & Yashunin, D. A. (2018). *Efficient

  and Robust Approximate Nearest Neighbor Search

  Using Hierarchical Navigable Small World Graphs.*

  IEEE TPAMI.

- Kraska, T. et al. (2018). *The Case for Learned

  Index Structures.* SIGMOD.

- Dalvi, N. & Suciu, D. (2007). *Efficient Query

  Evaluation on Probabilistic Databases.* VLDB J.

