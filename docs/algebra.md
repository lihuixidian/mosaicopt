# Algebra: Syntax, Semantics, and Equational Theory

> This document is the formalization of the MosaicOpt extended algebra. It corresponds to **Sections 3 and 4** of the project paper. The design context — the operator set, predicate families, and motivating claims — is the design brief at `docs/brief.md`. The formal material here is the substrate contract against which the rule-based optimizer (Section 6) and the cost model (Section 7) are constructed.

---

## 3. The Extended Algebra: Syntax and Semantics

### 3.1 Overview

This section formalizes the **logical algebra** that forms the substrate contract of MosaicOpt. The algebra extends classical relational algebra (Codd, 1970) with four new operators — **materialization** $\mu$, **ordering** $\tau$, **limit** $\Lambda$, and **graph expansion** $\Gamma$ — and four new predicate forms (vector range, BM25, graph reachability, graph edge) that capture the multi-modal query needs of agentic memory. The inherited operators ($\sigma$, $\pi$, $\bowtie$, $\cup$, $\cap$, $-$, $\rho$) are stated in the standard way; the four new operators, and the new predicate forms, are the genuinely novel content.

Two design decisions are made explicit up front:

1. **Bag (multiset) semantics.** The algebra operates on bags rather than sets, matching the de-facto semantics of SQL. Deduplication (`SELECT DISTINCT`) is recoverable via an explicit $\pi$ over the full schema, but the absence of an explicit DISTINCT in a query is *not* an instruction to deduplicate. Bag semantics also interact cleanly with physical operators such as HNSW range scans that naturally produce multiset output.
2. **Type discipline.** Every operator has a well-formedness pre-condition (schema containment, type compatibility, freshness of new attribute names). Ill-formed expressions are statically rejected.

Section 3.2 fixes notation, Section 3.3 gives the BNF grammar, and Section 3.4 gives the formal semantics of each operator.

### 3.2 Preliminaries: Schemas, Tuples, and Bags

Let $\mathcal{A}$ be a countably infinite set of **attribute names**. A **schema** is a finite subset $S \subseteq \mathcal{A}$. Each attribute $a \in S$ is associated with a **type** $\tau(a) \in \{\text{int}, \text{float}, \text{string}, \text{bool}, \text{vector}, \text{text}, \text{node\_id}, \ldots\}$. The typing discipline enforces: vector-typed predicates may only target vector-typed attributes; text-typed predicates may only target text-typed attributes; graph predicates operate on `node_id`-typed attributes; classical predicates are defined for scalar types.

A **tuple** over schema $S$ is a total function $t : S \to \mathcal{V}$ where $\mathcal{V}$ is the disjoint union of value domains (one per type), with $t(a) \in \mathcal{V}_{\tau(a)}$ for each $a \in S$. We write $T_S$ for the set of all tuples over $S$, and $t.a$ for the value of $t$ at attribute $a$. For $A \subseteq S$, $\pi_A(t)$ is the projection of $t$ to schema $A$, defined as the restriction $t|_A$.

A **bag** (multiset) over schema $S$ is a function $B : T_S \to \mathbb{N}$ with finite support (i.e., $B(t) > 0$ for finitely many $t$). The value $B(t)$ is the **multiplicity** of $t$ in $B$. The **cardinality** is $|B| = \sum_{t \in T_S} B(t)$. The set of tuples *underlying* $B$ is $\text{set}(B) = \{t : B(t) > 0\}$.

We write $\mathcal{B}_S$ for the set of all bags over $S$, and overload $\in$ to mean membership in the underlying set: $t \in B$ iff $B(t) > 0$.

An **ordered bag** (sorted list) over $S$ is a total order on a bag: a function $L : \mathbb{N}_{< n} \to T_S$ for some $n$, together with the underlying bag $B_L$ where $B_L(L(i)) = |\{j : L(j) = L(i)\}|$. We write $\mathcal{O}_S$ for the set of ordered bags. The operator $\tau$ (Section 3.4.8) produces elements of $\mathcal{O}_S$.

### 3.3 Syntax

#### 3.3.1 BNF Grammar

The grammar is parameterized by disjoint syntactic categories: identifiers (for relation and attribute names), constants (for literals), and query-bound parameters (for values supplied at query time).

```
<query>      ::=  <expr>                                     -- a query is an expression

<expr>       ::=  <ident>                                   -- base relation
              |   σ_<pred>(<expr>)                           -- selection
              |   π_<attrset>(<expr>)                        -- projection
              |   μ_<mexpr>→<ident>(<expr>)                 -- materialization
              |   <expr> ⋈_<pred> <expr>                     -- theta-join
              |   <expr> ∪ <expr>                            -- bag union
              |   <expr> ∩ <expr>                            -- bag intersection
              |   <expr> − <expr>                            -- bag difference
              |   ρ_<ident>→<ident>(<expr>)                 -- rename
              |   τ_<oexpr>(<expr>)                          -- ordering
              |   Λ_<int>(<expr>)                            -- limit
              |   Γ_<ident>,<int>(<expr>)                    -- graph expansion

<pred>      ::=  <atom>                                     -- atomic predicate
              |   ¬<pred>                                   -- negation
              |   <pred> ∧ <pred>                           -- conjunction
              |   <pred> ∨ <pred>                           -- disjunction

<atom>      ::=  <ident> <cmpop> <const>                    -- classical: a θ c
              |   <ident> ∈ <setconst>                      -- set membership
              |   ‖<ident> − <qparam>‖ <cmpop> <const>      -- vector range
              |   BM25(<ident>, <qparam>) <cmpop> <const>   -- text relevance
              |   reachable(<qparam>, <ident>)              -- graph reachability
              |   edge(<qparam>, <qparam>, <ident>)         -- labeled edge

<mexpr>     ::=  <const>                                    -- constant
              |   <ident>                                   -- attribute reference
              |   ‖<ident> − <qparam>‖                      -- vector distance
              |   BM25(<ident>, <qparam>)                   -- BM25 score
              |   (<mexpr>)                                 -- parenthesized
              |   <mexpr> <arithop> <mexpr>                 -- arithmetic
              |   min(<mexpr>, <mexpr>) | max(<mexpr>, <mexpr>)  -- extremum

<oexpr>     ::=  <mexpr> [ASC | DESC]                       -- single order key
              |   <oexpr>, <oexpr>                          -- lexicographic composition

<attrset>   ::=  <ident> | <ident> ∪ <attrset>              -- explicit union for sets

<cmpop>     ::=  = | ≠ | < | ≤ | > | ≥
<arithop>   ::=  + | − | * | /
<qparam>    ::=  ?<ident>                                   -- query-bound parameter
<setconst>  ::=  { <const>, <const>, ... }                  -- literal set
<ident>     ::=  [A-Za-z_][A-Za-z0-9_]*
<const>     ::=  numeric literal | string literal | NULL
<int>       ::=  positive integer literal
```

**Well-formedness preconditions** (enforced statically):

| Operator | Precondition |
|---|---|
| $\sigma_\phi(R)$ | $R : \mathcal{B}_S$; $\phi$ is a predicate over schema $S$ |
| $\pi_A(R)$ | $R : \mathcal{B}_S$; $A \subseteq S$ |
| $\mu_{e \to c}(R)$ | $R : \mathcal{B}_S$; $c \notin S$ (fresh); $e$ is evaluable in $S$ |
| $R \bowtie_\psi S$ | $R : \mathcal{B}_{S_R}$, $S : \mathcal{B}_{S_S}$; $\psi$ is a predicate over $S_R \cup S_S$ |
| $R \cup S$, $R \cap S$, $R - S$ | $R : \mathcal{B}_S$, $S : \mathcal{B}_S$ (same schema) |
| $\rho_{a \to b}(R)$ | $R : \mathcal{B}_S$; $a \in S$; $b \notin S$ |
| $\tau_O(R)$ | $R : \mathcal{B}_S$ |
| $\Lambda_K(R)$ | $R : \mathcal{O}_S$ (ordered bag) |
| $\Gamma_{g, d}(R)$ | $R : \mathcal{B}_S$; $S$ has a distinguished `node_id` attribute; $g$ is a named graph rule; $d \in \mathbb{N}^+$ |

The fresh-name preconditions ($\mu$ and $\rho$) prevent ambiguous references; the same-name preconditions for $\rho$ and join are conventional.

#### 3.3.2 Query Parameter Binding

The grammar contains two kinds of identifiers: compile-time identifiers (`<ident>`) and query-bound parameters (`<qparam>`, written `?name`). A query-bound parameter is a **hole** — a placeholder for a value supplied at query time. The algebra is *parameterized* by the values bound to the parameters; the semantics are defined for any well-typed binding.

Two consequences follow:

1. **Predicate distinguishability.** The expression $\sigma_{\|V_a - q\| < \tau}(R)$ is parametric in $q$ and $\tau$. Two different bindings of $q$ produce two different logical operators. A planner that specializes the operator for a particular $q$ is making a *physical* decision (e.g., choosing a specific HNSW entry point), not a *logical* one.
2. **Equivalence is parameter-uniform.** The identities in Section 4 hold for *all* bindings, with the proviso that identities referencing the *same* parameter on both sides require the bindings to match (as in J1, where the $q$ on both sides refers to the same parameter).

### 3.4 Semantics (Bag Interpretation)

We give the semantics of each operator as a function from input bags (or ordered bags) to output bags (or ordered bags). For each operator, we state the type of the operator, the well-formedness pre-condition, and the output schema.

#### 3.4.1 Base Relation

A base relation is a stored bag identified by a name $R$. Its bag is the contents of the relation in the catalog at query time, and its schema is the catalog schema.

$$\llbracket R \rrbracket = B_R \in \mathcal{B}_{\text{schema}(R)}$$

The multiplicity function is given by the storage engine (sequential scan, index, etc.).

#### 3.4.2 Selection ($\sigma$)

**Type:** $\mathcal{B}_S \to \mathcal{B}_S$ (predicate $\phi$ is a Boolean function on $T_S$).

**Semantics:**
$$[\![\sigma_\phi(R)]\!](t) \;=\; \begin{cases} [\![R]\!](t) & \text{if } \phi(t) = \text{true} \\ 0 & \text{otherwise} \end{cases}$$

The selection preserves the schema and the multiplicity of each surviving tuple.

**Predicate evaluation.** A predicate $\phi$ evaluates to true or false for each tuple. For the novel predicate forms:

- **Vector range** $\phi \equiv \|V_a - q\| < \tau$: $V_a$ is a vector-typed attribute; $q \in \mathcal{V}_{\text{vector}}$ is a query-bound parameter; $\tau \in \mathbb{R}^+$. The evaluation is the standard distance comparison.
- **BM25** $\phi \equiv \text{BM25}(t_a, q) > \theta$: $t_a$ is a text-typed attribute; $q$ is a query-bound string; $\theta \in \mathbb{R}^+$. The BM25 score is computed against the text column statistics (document frequencies, average document length) of the relation.
- **Reachability** $\phi \equiv \text{reachable}(q, a)$: $a$ is a `node_id`-typed attribute; $q$ is a query-bound node id. True iff $q$ is in the same connected component as any node reachable from $a$ via the default graph schema.
- **Edge** $\phi \equiv \text{edge}(q_1, q_2, e)$: $q_1$, $q_2$ are query-bound node ids; $e$ is a label (or attribute reference). True iff a directed edge labeled $e$ exists from $q_1$ to $q_2$ in the graph catalog.

#### 3.4.3 Projection ($\pi$)

**Type:** $\mathcal{B}_S \to \mathcal{B}_A$ (with $A \subseteq S$).

**Semantics:**
$$[\![\pi_A(R)]\!](t') \;=\; \sum_{t \in T_S \,:\, \pi_A(t) = t'} [\![R]\!](t)$$

Projection may collapse multiplicities: if two distinct tuples $t_1, t_2 \in T_S$ project to the same $t' \in T_A$, the multiplicity of $t'$ in $\pi_A(R)$ is the sum $[\![R]\!](t_1) + [\![R]\!](t_2)$.

#### 3.4.4 Materialization ($\mu$)

**Type:** $\mathcal{B}_S \to \mathcal{B}_{S \cup \{c\}}$ (with $c \notin S$).

**Semantics:** Materialization evaluates an expression $e$ for each tuple of the input bag and appends the result as a new attribute $c$:

$$[\![\mu_{e \to c}(R)]\!](t \oplus (c = v)) \;=\; \begin{cases} [\![R]\!](t) & \text{if } v = [\![e]\!](t) \\ 0 & \text{otherwise} \end{cases}$$

where $[\![e]\!](t)$ is the evaluation of expression $e$ on tuple $t$. The output schema is $S \cup \{c\}$; the multiplicity of each input tuple is preserved exactly once (one output tuple per input tuple, with the new column appended).

**Critical property.** Materialization exposes computed values as first-class attributes. A subsequent operator can reference $c$ directly — most importantly, an `ORDER BY` clause can order on $c$, and a subsequent $\sigma$ can filter on $c$. This is the design choice that makes score-fusion policies expressible as SQL `ORDER BY` expressions rather than baked into the storage substrate.

**Expression evaluation.** The expression language for $e$ includes:
- arithmetic over scalar attributes: $a + b$, $a \cdot 2$, etc.;
- vector distance: $\|V_a - q\|$, which yields a scalar (the L2 norm of the difference);
- BM25: $\text{BM25}(t_a, q)$, which yields a real-valued score;
- compositions: $\alpha \cdot \|V_a - q\| + \beta \cdot \text{BM25}(t_b, q')$, etc.

The type discipline is strict: a vector expression $\|V_a - q\|$ is only well-typed when $V_a$ is vector-typed and $q$ is a query-bound vector.

#### 3.4.5 Join ($\bowtie$)

**Type:** $\mathcal{B}_{S_R} \times \mathcal{B}_{S_S} \to \mathcal{B}_{S_R \cup S_S}$ (theta-join, with the convention that $S_R$ and $S_S$ are disjoint; attribute clashes are resolved by $\rho$ before the join).

**Semantics:** The theta-join is a cross product filtered by the join predicate, with multiplicities multiplied:

$$[\![R \bowtie_\psi S]\!](t) \;=\; \sum_{\substack{t_R, t_S \,:\, t_R \oplus t_S = t \\ \psi(t_R, t_S)}} [\![R]\!](t_R) \cdot [\![S]\!](t_S)$$

The natural join $R \bowtie S$ is the special case $\psi \equiv \bigwedge_{a \in S_R \cap S_S} (t_R.a = t_S.a)$, with $S_R \cap S_S$ retained once in the output.

#### 3.4.6 Set Operations ($\cup$, $\cap$, $-$)

**Type:** $\mathcal{B}_S \times \mathcal{B}_S \to \mathcal{B}_S$ (both operands over the same schema).

**Semantics (bag semantics):**

$$[\![R \cup S]\!](t) = [\![R]\!](t) + [\![S]\!](t)$$
$$[\![R \cap S]\!](t) = \min\bigl([\![R]\!](t),\, [\![S]\!](t)\bigr)$$
$$[\![R - S]\!](t) = \max\bigl([\![R]\!](t) - [\![S]\!](t),\, 0\bigr)$$

These follow the SQL semantics of the corresponding operators (in particular, `EXCEPT` is bag difference, not set difference).

#### 3.4.7 Rename ($\rho$)

**Type:** $\mathcal{B}_S \to \mathcal{B}_{S'}$ (with $a \in S$, $b \notin S$, $S' = (S \setminus \{a\}) \cup \{b\}$).

**Semantics:**
$$[\![\rho_{a \to b}(R)]\!](t') \;=\; \sum_{t \,:\, \rho_{a \to b}(t) = t'} [\![R]\!](t)$$

where $\rho_{a \to b}(t)$ is the tuple $t$ with attribute $a$ renamed to $b$. (In particular, $t'$ is over the schema $S'$ and the rename is a pure relabeling.)

#### 3.4.8 Ordering ($\tau$)

**Type:** $\mathcal{B}_S \to \mathcal{O}_S$ (bag to ordered bag).

**Semantics:** $\tau_O(R)$ returns the elements of $R$ as an ordered list sorted by the order expression $O$. The default direction is ascending (`ASC`); `DESC` reverses.

Concretely, $[\![\tau_O(R)]\!]$ is the unique ordered list $L$ of length $|[\![R]\!]|$ such that:
1. the underlying bag of $L$ equals $[\![R]\!]$ (same multiset of tuples);
2. for all $i < j < |L|$, $[\![O]\!](L(i)) \le [\![O]\!](L(j))$ (sorted by $O$ ascending);
3. the sort is **stable**: tuples with equal $O$-value preserve their input order.

For a compound order expression $O_1, O_2$ (lexicographic), the sort is first by $O_1$, then by $O_2$ as a tie-breaker.

**The "sorted bag" type is significant downstream.** The limit operator $\Lambda_K$ is the only operator that requires an ordered input. The convention in this algebra is that $\tau$ and $\Lambda$ are paired: the typical use is $\Lambda_K(\tau_O(R))$ to express "top-K by $O$".

#### 3.4.9 Limit ($\Lambda$)

**Type:** $\mathcal{O}_S \to \mathcal{B}_S$ (ordered bag to bag).

**Semantics:** $\Lambda_K(R)$ returns the first $K$ elements of the ordered list $R$ (or all of $R$ if $|R| \le K$):

$$[\![\Lambda_K(R)]\!](t) \;=\; |\{i < \min(K, |R|) : R(i) = t\}|$$

That is, the multiplicity of $t$ in $\Lambda_K(R)$ is the number of times $t$ appears in the prefix of $R$ of length $K$.

**Convention on bare use.** In the well-formed fragment of the algebra, $\Lambda_K$ consumes an ordered bag (typically the output of $\tau_O$). A direct application $\Lambda_K(R)$ where $R$ is an unordered bag is permitted but produces a non-deterministic result (any $K$-element sub-bag of $R$). The T1 identity (Section 4.4.6) covers the canonical ordered-input case.

#### 3.4.10 Graph Expansion ($\Gamma$)

**Type:** $\mathcal{B}_S \to \mathcal{B}_{S \cup \{\text{target}, \text{path}\}}$ (with a distinguished `node_id` attribute in $S$ and a graph rule $g$ of fixed arity).

**Semantics.** A **graph rule** $g$ is a named binary relation over `node_id`-typed values: $E_g \subseteq \mathcal{V}_{\text{node\_id}} \times \mathcal{V}_{\text{node\_id}}$ (the extension of $g$ in the graph catalog). A **path** of length $\ell$ from node $u$ to node $v$ under rule $g$ is a sequence of $\ell$ edges $(u_0, u_1), (u_1, u_2), \ldots, (u_{\ell-1}, u_\ell) \in E_g$ with $u_0 = u$ and $u_\ell = v$.

For an input bag $R$ (whose schema $S$ has a distinguished `node_id` attribute $n$), $\Gamma_{g, d}(R)$ produces, for each tuple $t \in R$ and each path $p$ from $n(t)$ of length at most $d$ under $g$, an output tuple consisting of the original attributes of $t$ (unchanged), plus a `target` attribute holding the path's terminal node and a `path` attribute holding the sequence of edges.

Formally, let $\Pi_{g, d}(u)$ be the set of all (terminal node, path) pairs reachable from $u$ in at most $d$ steps under $g$:

$$\Pi_{g, d}(u) = \{(v, p) : \text{$p$ is a path of length $\le d$ from $u$ to $v$ under $g$}\}$$

Then:

$$[\![\Gamma_{g, d}(R)]\!](t \oplus (\text{target} = v, \text{path} = p)) = \sum_{\substack{t \in \text{set}([\![R]\!]) \\ (v, p) \in \Pi_{g, d}(t.n)}} [\![R]\!](t)$$

The output schema is $S \cup \{\text{target}, \text{path}\}$. The multiplicity of an input tuple $t$ is multiplied by $|\Pi_{g, d}(t.n)|$ (the number of distinct reachable (target, path) pairs from $t$).

**Design choice (path-preserving).** The expansion preserves path information: each output tuple records the seed tuple $t$, the target node $v$, and the path $p$ that led from $t$ to $v$. This is the more expressive variant (vs. a path-collapsing variant that returns only the target nodes) and supports downstream provenance-aware operators. The trade-off is a larger output schema and a `path` attribute of variable type (a list of edges); the physical algebra (Section 4.2 of the brief) addresses the storage overhead.

**Empty case.** If $|\Pi_{g, d}(t.n)| = 0$ (no reachable node from $t.n$ in $\le d$ steps), then $t$ contributes no output tuples. This is *not* the same as a "no-match" inner join — seeds with no reachable nodes are silently dropped.

---

## 4. Equational Theory

### 4.1 Overview

This section develops the **equational theory** of the extended algebra. Equational theory is the substrate of the rule-based optimizer: each identity is a rewrite rule that transforms one logical plan into an equivalent one, expanding (or contracting) the plan space that the cost-based optimizer must consider. The identities in this section are the rewrite rules implemented in the RBO.

We organize the equational theory into three layers:

1. **Inherited foundational axioms** (Section 4.2) — the standard relational algebra identities that the extended algebra continues to satisfy, stated as axioms without proof (proofs are in the relational algebra literature).
2. **New foundational axioms** (Section 4.3) — the definitional identities that govern the four new operators ($\mu$, $\tau$, $\Lambda$, $\Gamma$).
3. **Derived identities** (Section 4.4) — non-trivial identities whose proofs are the main technical content. The proofs are algebraic: each proof applies the foundational axioms and unfolds operator definitions to reduce both sides of the equation to a common form.

We also catalog the **non-identities** (Section 4.5) — pairs of expressions that look like they should be equal but are not, with explicit counterexamples. The non-identities are pedagogically important: they identify the points where the logical algebra and the physical algebra diverge, which is exactly where the logical/physical separation earns its keep.

### 4.2 Inherited Foundational Axioms

The following identities are inherited from relational algebra. They hold for the extended algebra under the well-formedness preconditions of Section 3.3.1. Proofs appear in any standard text (e.g., Ullman, 1988; Garcia-Molina et al., 2008); we do not reproduce them.

**A1 (Selection composition).**
$$\sigma_P(\sigma_Q(R)) \;=\; \sigma_{P \wedge Q}(R)$$

**A2 (Selection distributes over union).**
$$\sigma_P(R \cup S) \;=\; \sigma_P(R) \cup \sigma_P(S)$$

**A3 (Projection idempotence).**
$$\pi_A(\pi_B(R)) \;=\; \pi_A(R) \qquad \text{when } A \subseteq B$$

(More generally, $\pi_A(\pi_B(R)) = \pi_{A \cap B}(R)$.)

**A4 (Selection pushdown through join).**
$$\sigma_P(R \bowtie_\psi S) \;=\; \sigma_P(R) \bowtie_\psi S \qquad \text{when all attributes referenced by } P \text{ are in } \text{attr}(R)$$

These four axioms are the workhorses of classical RBO; the derived identities of Section 4.4 are largely specializations to the new predicate and operator forms.

### 4.3 New Foundational Axioms

The new foundational axioms govern the four new operators. Unlike the inherited axioms, these are largely *definitional*: each axiom states an immediate consequence of the operator's semantics in Section 3.4.

**Materialization axioms.**

- **MA1** (schema extension): $\text{attr}(\mu_{e \to c}(R)) = \text{attr}(R) \cup \{c\}$.
- **MA2** (column identity): for $t \in \text{set}([\![R]\!])$, the new column $c$ of the corresponding output tuple equals $[\![e]\!](t)$.
- **MA3** (multiplicity preservation): for $t \in T_{\text{attr}(R)}$, the multiplicity of $(t, c = [\![e]\!](t))$ in $\mu_{e \to c}(R)$ is $[\![R]\!](t)$.

**Ordering axioms.**

- **OA1** (sort stability): $\tau_O(R)$ is the stable sort of $R$ by $O$.
- **OA2** (ordering idempotence): $\tau_O(\tau_O(R)) = \tau_O(R)$.
- **OA3** (lexicographic decomposition): $\tau_{O_1, O_2}(R) = \tau_{O_1}(\tau_{O_2}(R))$ when $O_1$ and $O_2$ are independent (i.e., $O_1$ does not reference columns added by $O_2$'s evaluation). (We will not need the general form below.)

**Limit axioms.**

- **LA1** (limit on an already-short input): if $|R| \le K$, then $\Lambda_K(R) = R$ as an ordered bag.
- **LA2** (limit on a zero-length input): $\Lambda_K(\emptyset) = \emptyset$.
- **LA3** (limit type): $\Lambda_K$ consumes an ordered bag; its output is a (possibly shorter) ordered bag, or equivalently an unordered bag (the order is irrelevant past the prefix).

**Graph expansion axioms.**

- **GA1** (schema extension): $\text{attr}(\Gamma_{g, d}(R)) = \text{attr}(R) \cup \{\text{target}, \text{path}\}$.
- **GA2** (monotonicity in $d$): for $d_1 \le d_2$, $\text{set}(\Gamma_{g, d_1}(R)) \subseteq \text{set}(\Gamma_{g, d_2}(R))$ (under the natural embedding via the prefix-of-path relation).
- **GA3** (no expansion at $d=0$): $\Gamma_{g, 0}(R)$ is the empty bag — the only path of length 0 is the trivial path from a node to itself, and the convention here is to return no tuples for $d=0$. (This is a design choice; the alternative is to return the seed tuples with `target = seed, path = ()`.)

These are not the only identities one could state; they are the subset invoked by the proofs in Section 4.4.

### 4.4 Derived Identities

The identities in this section are the main technical content of the equational theory. Each is a derived identity: it follows from the foundational axioms (Section 4.3) and the inherited axioms (Section 4.2), plus unfolding the operator semantics of Section 3.4. The proofs are short but careful.

#### 4.4.1 M1 — Selection Commutes with Materialization on the Materialized Column

> **M1.** $\sigma_{c < \tau}\bigl(\mu_{e \to c}(R)\bigr) \;=\; \mu_{e \to c}\bigl(\sigma_{e < \tau}(R)\bigr)$, where $c \notin \text{attr}(R)$ is fresh.

**Proof.** Let $S = \text{attr}(R)$. We compute each side as a function on $T_{S \cup \{c\}}$, and show the multiplicity functions are identical.

*Left-hand side.* For $t' = t \oplus (c = v) \in T_{S \cup \{c\}}$:

$$[\![\sigma_{c < \tau}(\mu_{e \to c}(R))]\!](t') = \begin{cases} [\![\mu_{e \to c}(R)]\!](t') & \text{if } v < \tau \\ 0 & \text{otherwise} \end{cases}$$

By MA2, $[\![\mu_{e \to c}(R)]\!](t') = [\![R]\!](t)$ when $v = [\![e]\!](t)$, and 0 otherwise. Substituting:

$$[\![\sigma_{c < \tau}(\mu_{e \to c}(R))]\!](t') = \begin{cases} [\![R]\!](t) & \text{if } v = [\![e]\!](t) \text{ and } v < \tau \\ 0 & \text{otherwise} \end{cases}$$

Note that $v = [\![e]\!](t) \wedge v < \tau$ is equivalent to $v = [\![e]\!](t) \wedge [\![e]\!](t) < \tau$ (since $v$ is constrained to equal $[\![e]\!](t)$ in the positive case). We can therefore write the condition more compactly as: $v = [\![e]\!](t)$ and $[\![e]\!](t) < \tau$.

*Right-hand side.* For $t \in T_S$:

$$[\![\sigma_{e < \tau}(R)]\!](t) = \begin{cases} [\![R]\!](t) & \text{if } [\![e]\!](t) < \tau \\ 0 & \text{otherwise} \end{cases}$$

Applying $\mu_{e \to c}$ (MA2, MA3):

$$[\![\mu_{e \to c}(\sigma_{e < \tau}(R))]\!](t \oplus (c = v)) = \begin{cases} [\![\sigma_{e < \tau}(R)]\!](t) & \text{if } v = [\![e]\!](t) \\ 0 & \text{otherwise} \end{cases}$$

$$= \begin{cases} [\![R]\!](t) & \text{if } v = [\![e]\!](t) \text{ and } [\![e]\!](t) < \tau \\ 0 & \text{otherwise} \end{cases}$$

The positive-case conditions on the two sides are identical: $v = [\![e]\!](t) \wedge v < \tau$ vs. $v = [\![e]\!](t) \wedge [\![e]\!](t) < \tau$. These are the same condition (the variable $v$ only appears in the equality constraint $v = [\![e]\!](t)$, under which $v < \tau$ is equivalent to $[\![e]\!](t) < \tau$). Therefore the multiplicity functions agree on all $t' \in T_{S \cup \{c\}}$, and the bags are equal. $\blacksquare$

**Significance for the optimizer.** M1 enables the rewrite

$$\sigma_{c < \tau}(\mu_{e \to c}(R)) \;\Longrightarrow\; \mu_{e \to c}(\sigma_{e < \tau}(R))$$

which fuses the selection with the materialization: instead of computing $e$ for every tuple and then filtering, the optimizer computes $e$ only on tuples that pass the filter. The saving scales with the selectivity of $e < \tau$ — substantial for tight vector ranges.

#### 4.4.2 M2 — Materialization Commutativity

> **M2.** $\mu_{e_1 \to c_1}\bigl(\mu_{e_2 \to c_2}(R)\bigr) \;=\; \mu_{e_2 \to c_2}\bigl(\mu_{e_1 \to c_1}(R)\bigr)$, where $c_1, c_2 \notin \text{attr}(R)$ are fresh and distinct, and $e_1, e_2$ are *independent*: $e_1$ does not reference $c_2$ and $e_2$ does not reference $c_1$.

**Proof.** Let $S = \text{attr}(R)$. Both sides have schema $S \cup \{c_1, c_2\}$.

*Left-hand side.* First apply $\mu_{e_2 \to c_2}$ (MA2, MA3):

$$[\![\mu_{e_2 \to c_2}(R)]\!](t \oplus (c_2 = v_2)) = \begin{cases} [\![R]\!](t) & \text{if } v_2 = [\![e_2]\!](t) \\ 0 & \text{otherwise} \end{cases}$$

Then apply $\mu_{e_1 \to c_1}$. By independence, $e_1$ does not reference $c_2$, so $[\![e_1]\!](t \oplus (c_2 = v_2)) = [\![e_1]\!](t)$:

$$[\![\mu_{e_1 \to c_1}(\mu_{e_2 \to c_2}(R))]\!](t \oplus (c_1 = v_1, c_2 = v_2)) = \begin{cases} [\![R]\!](t) & \text{if } v_1 = [\![e_1]\!](t) \text{ and } v_2 = [\![e_2]\!](t) \\ 0 & \text{otherwise} \end{cases}$$

*Right-hand side.* By the symmetric argument (now $e_2$ does not reference $c_1$, so $[\![e_2]\!](t \oplus (c_1 = v_1)) = [\![e_2]\!](t)$):

$$[\![\mu_{e_2 \to c_2}(\mu_{e_1 \to c_1}(R))]\!](t \oplus (c_1 = v_1, c_2 = v_2)) = \begin{cases} [\![R]\!](t) & \text{if } v_1 = [\![e_1]\!](t) \text{ and } v_2 = [\![e_2]\!](t) \\ 0 & \text{otherwise} \end{cases}$$

The two multiplicity functions are identical on $T_{S \cup \{c_1, c_2\}}$. (The column ordering $c_1, c_2$ vs. $c_2, c_1$ is immaterial for bag equality — we identify tuples up to column permutation under the shared schema.) $\blacksquare$

**Note on the independence precondition.** The independence condition is essential. If $e_2$ referenced $c_1$, then the two sides would compute $c_2$ against different inputs (in the LHS, $c_2 = [\![e_2]\!](t, c_1 = [\![e_1]\!](t))$; in the RHS, $c_2 = [\![e_2]\!](t)$ alone), and the bags would differ. M2 is a rewrite rule for independent materializations only.

**Significance for the optimizer.** M2 enables materialization reordering: the optimizer can choose the order in which computed columns are appended, e.g., to interleave with a filter. M1 + M2 together permit pushing a selective materialization past a non-selective one, saving the cost of computing the non-selective expression for filtered-out rows.

#### 4.4.3 M3 — Materialization Through Projection

> **M3.** $\mu_{e \to c}\bigl(\pi_A(R)\bigr) \;=\; \pi_{A \cup \{c\}}\bigl(\mu_{e \to c}(R)\bigr)$, where $c \notin \text{attr}(R)$ is fresh (hence $c \notin A$) and $e$ references only attributes in $A$.

**Proof.** Let $S = \text{attr}(R)$. Note that $A \subseteq S$ and $c \notin A$.

*Left-hand side.* First project $R$ to $A$ (Section 3.4.3):

$$[\![\pi_A(R)]\!](t') = \sum_{t \in T_S \,:\, \pi_A(t) = t'} [\![R]\!](t) \qquad (t' \in T_A)$$

Then materialize $c$. By the precondition that $e$ references only attributes in $A$, the expression $e$ is well-typed against $T_A$, and $[\![e]\!](t') = [\![e]\!](t)$ for any $t$ with $\pi_A(t) = t'$. Applying MA2, MA3:

$$[\![\mu_{e \to c}(\pi_A(R))]\!](t' \oplus (c = v)) = \begin{cases} \sum_{t \,:\, \pi_A(t) = t'} [\![R]\!](t) & \text{if } v = [\![e]\!](t') \\ 0 & \text{otherwise} \end{cases}$$

*Right-hand side.* First materialize $c$ on $R$ (MA2, MA3):

$$[\![\mu_{e \to c}(R)]\!](t \oplus (c = v)) = \begin{cases} [\![R]\!](t) & \text{if } v = [\![e]\!](t) \\ 0 & \text{otherwise} \end{cases}$$

Then project to $A \cup \{c\}$. Since $c \notin A$, the projection retains all attributes:

$$[\![\pi_{A \cup \{c\}}(\mu_{e \to c}(R))]\!](t' \oplus (c = v)) = \sum_{\substack{t \in T_S \\ t' = \pi_A(t) \\ v = [\![e]\!](t)}} [\![R]\!](t)$$

This sum ranges over all $t \in T_S$ with $\pi_A(t) = t'$ and $[\![e]\!](t) = v$. Using the precondition $[\![e]\!](t) = [\![e]\!](t')$ (since $e$ only references attributes in $A$), the condition $[\![e]\!](t) = v$ is equivalent to $[\![e]\!](t') = v$. Therefore:

$$[\![\pi_{A \cup \{c\}}(\mu_{e \to c}(R))]\!](t' \oplus (c = v)) = \begin{cases} \sum_{t \,:\, \pi_A(t) = t'} [\![R]\!](t) & \text{if } v = [\![e]\!](t') \\ 0 & \text{otherwise} \end{cases}$$

The two sides agree on every tuple. $\blacksquare$

**Precondition discussion.** The "$e$ references only attributes in $A$" precondition is required: if $e$ referenced $b \in S \setminus A$, then $\pi_A(R)$ would not carry $b$ and $e$ would be ill-typed against it. M3 is the projection-fusion dual of M1.

#### 4.4.4 V1 — Vector Selections on Distinct Columns Commute

> **V1.** $\sigma_{\|V_a - q_a\| < \tau_a}\bigl(\sigma_{\|V_b - q_b\| < \tau_b}(R)\bigr) \;=\; \sigma_{\|V_b - q_b\| < \tau_b}\bigl(\sigma_{\|V_a - q_a\| < \tau_a}(R)\bigr)$, where $V_a \ne V_b$ are vector-typed attributes of $R$ and $q_a, q_b$ are query-bound vectors.

**Proof.** This identity is an immediate consequence of the selection-composition axiom (A1) and the commutativity of $\wedge$:

*Left-hand side.* By A1:

$$\sigma_{\|V_a - q_a\| < \tau_a}\bigl(\sigma_{\|V_b - q_b\| < \tau_b}(R)\bigr) = \sigma_{(\|V_a - q_a\| < \tau_a) \wedge (\|V_b - q_b\| < \tau_b)}(R)$$

*Right-hand side.* Similarly:

$$\sigma_{\|V_b - q_b\| < \tau_b}\bigl(\sigma_{\|V_a - q_a\| < \tau_a}(R)\bigr) = \sigma_{(\|V_b - q_b\| < \tau_b) \wedge (\|V_a - q_a\| < \tau_a)}(R)$$

Since $\wedge$ is commutative, the two predicates are equivalent: $(P \wedge Q) = (Q \wedge P)$ for any two predicates $P, Q$. The two $\sigma$-expressions are therefore the same operator. $\blacksquare$

**Significance — logical, not physical.** V1 is a **logical** identity: the two plans produce identical output bags. It does **not** imply equal cost — the two orderings may differ by orders of magnitude at the physical layer, where $\sigma_{\|V_a - q_a\| < \tau_a}$ may be implemented by an HNSW probe whose cost depends on the radius, the index size, and `efSearch`. V1's role is to **expand the plan space**: the optimizer may choose either order, plus any further transformations, before the cost model commits.

**A direct corollary — the multi-vector range query.** V1 generalizes to $n$ vector predicates on distinct columns:

$$\sigma_{\|V_1 - q_1\| < \tau_1} \cdots \sigma_{\|V_n - q_n\| < \tau_n}(R) \;=\; \sigma_{\bigwedge_{i=1}^n \|V_i - q_i\| < \tau_i}(R)$$

i.e., a multi-vector range query is a single conjunctive predicate — the substrate-level expression of "conjunctive semantic intent" across multiple vector columns (claim 1 of the brief).

#### 4.4.5 J1 — Vector Selection Pushdown Through Join

> **J1.** $\sigma_{\|V_a - q\| < \tau}(R \bowtie_\psi S) \;=\; \sigma_{\|V_a - q\| < \tau}(R) \bowtie_\psi S$, where $V_a \in \text{attr}(R)$ and $\psi$ is an arbitrary join predicate.

**Proof.** This identity is a direct application of the inherited axiom A4 (selection pushdown through join), specialized to the vector predicate. The precondition of A4 — that all attributes referenced by the selection predicate be in $\text{attr}(R)$ — is satisfied by hypothesis: $V_a \in \text{attr}(R)$ and $q$ is a query-bound parameter, not an attribute. By A4:

$$\sigma_P(R \bowtie_\psi S) = \sigma_P(R) \bowtie_\psi S \qquad \text{when } \text{attr}(P) \subseteq \text{attr}(R)$$

The identity follows immediately. $\blacksquare$

**Significance for the optimizer.** J1 enables the optimizer to push a vector range filter from the post-join output down to the pre-join scan of $R$. The saving is twofold: (1) the join processes fewer tuples, since many rows of $R$ are filtered out before the probe; (2) a selective filter may shrink the build side of a hash join. Together with V1, J1 is the principal rule by which multi-vector range queries are made efficient in MosaicOpt.

**Asymmetry note.** J1 pushes the filter to $R$ because the predicate only references $R$'s attributes. The dual identity (push to $S$) holds when the predicate only references $S$'s attributes. The asymmetry is semantic, not arbitrary.

#### 4.4.6 T1 — Order Plus Limit Is Top-K

> **T1.** $\tau_O\bigl(\Lambda_K(R)\bigr) \;=\; \mathrm{topK}_O(R, K)$, where $R$ is a bag and $O$ is a total order.

**Proof.** *Left-hand side.* By the convention of the algebra, the inner $\Lambda_K(R)$ is the limit applied to an *ordered* form of $R$: under the canonical use of $\tau$ and $\Lambda$ as a paired construct (Section 3.4.9), the expression $\tau_O(\Lambda_K(R))$ is short for $\Lambda_K(\tau_O(R))$ — the $K$ smallest elements of $R$ by $O$, in sorted order. To be precise about the identity: define $\mathrm{topK}_O(R, K) = \Lambda_K(\tau_O(R))$ as the *definition* of the top-K operation; the identity T1 is then a definitional equality.

To see that the definition is consistent: $\tau_O(R)$ is the sorted list of $R$ by $O$ (Section 3.4.8); $\Lambda_K$ of an ordered list returns its first $K$ elements (Section 3.4.9). The first $K$ elements of the sorted list are exactly the $K$ tuples of $R$ with the smallest $O$-values, with stable tie-breaking. Therefore:

$$\mathrm{topK}_O(R, K) = \Lambda_K(\tau_O(R))$$

and the identity T1 holds by definition. $\blacksquare$

**Significance — early termination.** T1 is the rewrite rule that makes top-K queries cheap. The right-hand side $\mathrm{topK}_O(R, K)$ admits a **heap-based implementation**: maintain a max-heap of size $K$ during a single scan of $R$, replacing the root whenever a smaller-$O$ tuple is encountered, then drain the heap in $O$-sorted order. This runs in $O(|R| \log K)$ time and $O(K)$ space. By contrast, evaluating $\tau_O(R)$ then $\Lambda_K$ requires a full sort in $O(|R| \log |R|)$ time. For small $K$ and large $|R|$, the difference is between a feasible plan and an infeasible one.

**The optimizer's use of T1.** The RBO recognizes the pattern $\tau_O(\Lambda_K(\cdot))$ and rewrites it to a single $\mathrm{topK}_O(\cdot, K)$ node; the CBO then chooses the heap-based physical implementation. The rewrites of M1, M2, and V1 may be applied *above* the topK node without disturbing early termination.

**Tie-breaking.** The definition requires $O$ to be a total order. In practice, $O$ is often a partial order (e.g., a distance value with ties); we extend by composing $O$ with a deterministic tie-breaker $T$ (e.g., lexicographic on remaining attributes or on the primary key), and the identity holds with $O$ replaced by the lexicographic composition $(O, T)$.

#### 4.4.7 G1 — Selection Pushdown Through Graph Expansion

> **G1.** $\sigma_P\bigl(\Gamma_{g, d}(R)\bigr) \;=\; \Gamma_{g, d}\bigl(\sigma_P(R)\bigr)$, where $P$ is a predicate that references only attributes in $\text{attr}(R)$ (the seed attributes, not `target` or `path`).

**Proof.** Let $S = \text{attr}(R)$. By the precondition, $P$ is over $S$. Let $\Pi_{g, d}(u)$ be the set of (target, path) pairs reachable from $u$ in at most $d$ steps (Section 3.4.10).

*Left-hand side.* Apply $\Gamma_{g, d}$ to $R$, then $\sigma_P$. For $t' \in T_{S \cup \{\text{target}, \text{path}\}}$:

$$[\![\Gamma_{g, d}(R)]\!](t') = \sum_{\substack{t \in \text{set}([\![R]\!]) \\ (v, p) \in \Pi_{g, d}(t.n) \\ t' = t \oplus (\text{target} = v, \text{path} = p)}} [\![R]\!](t)$$

Apply $\sigma_P$. Since $P$ references only $S$-attributes, $P(t') = P(t)$ for any $t' = t \oplus (\text{target} = v, \text{path} = p)$. The result:

$$[\![\sigma_P(\Gamma_{g, d}(R))]\!](t') = \begin{cases} \sum [\![R]\!](t) & \text{if $P(t)$ holds and $(v, p) \in \Pi_{g, d}(t.n)$ and $t' = t \oplus \ldots$} \\ 0 & \text{otherwise} \end{cases}$$

*Right-hand side.* Apply $\sigma_P$ to $R$ first: $[\![\sigma_P(R)]\!](t) = [\![R]\!](t)$ if $P(t)$, else 0. Then $\Gamma_{g, d}$:

$$[\![\Gamma_{g, d}(\sigma_P(R))]\!](t') = \sum_{\substack{t \in \text{set}([\![\sigma_P(R)]\!]) \\ (v, p) \in \Pi_{g, d}(t.n) \\ t' = t \oplus \ldots}} [\![\sigma_P(R)]\!](t)$$

The domain of the sum is restricted to $t$ with $P(t)$ holding. Expanding $[\![\sigma_P(R)]\!](t) = [\![R]\!](t)$ on this domain:

$$= \sum_{\substack{t \in \text{set}([\![R]\!]),\, P(t) \\ (v, p) \in \Pi_{g, d}(t.n) \\ t' = t \oplus \ldots}} [\![R]\!](t)$$

The two multiplicity functions agree on every $t' \in T_{S \cup \{\text{target}, \text{path}\}}$: in both cases, the contributing tuples are exactly those $t$ with $P(t)$ holding and $(v, p) \in \Pi_{g, d}(t.n)$ producing $t' = t \oplus (\text{target} = v, \text{path} = p)$. $\blacksquare$

**Significance for the optimizer.** G1 is the "seed filtering" optimization: a query that selects a property of the seed node, then expands the graph, can be rewritten so the filter is applied to seeds before expansion. The saving is substantial when the filter is selective, since only a fraction of seeds trigger a graph traversal.

**The dual case.** When $P$ references `target` (e.g., $\text{target} \in X$ for a set of target nodes), G1 does *not* apply — the filter must be applied after expansion. This is a useful negative result for the optimizer: it tells it which predicates can be pushed down.

### 4.5 Non-Identities

The identities of Section 4.4 are the rules the optimizer may apply. This section catalogs the **non-identities** — pairs of expressions that look like they should be equal but are not. Each non-identity is accompanied by a concrete counterexample, and by an explanation of *why* the two expressions differ, which is the pointer to where the logical/physical separation matters.

#### 4.5.1 N1 — HNSW Approximation Breaks Logical Equivalence

> **N1.** Logical $\sigma_{\|V - q\| < \tau}(R)$ is exactly defined. Physical $H_{V, q, \tau}(R)$ (HNSW range scan) is approximate. Therefore, the logical and physical operators are not equivalent in general:
> $$\sigma_{\|V - q\| < \tau}(R) \;\not\equiv\; H_{V, q, \tau}(R)$$

**Counterexample.** Let $R$ be a relation with schema $\{\text{id} : \text{int}, V : \text{vector}\}$ containing 10 tuples whose vectors are at positions $(0, 0), (1, 0), (2, 0), \ldots, (9, 0)$ in $\mathbb{R}^2$. Let $q = (5, 0)$ and $\tau = 3$.

*Logical evaluation.* $\sigma_{\|V - q\| < 3}(R)$ returns exactly the tuples with $\|V - (5,0)\| < 3$, i.e., the tuples at $(3, 0), (4, 0), (5, 0), (6, 0), (7, 0)$ — a set of 5 tuples. The evaluation is **exact**: every qualifying tuple is returned, and no non-qualifying tuple is returned.

*Physical evaluation.* $H_{V, q, 3}(R)$ using an HNSW index over $V$ with a low `efSearch` parameter (e.g., `efSearch = 2`) may return a **subset** of the 5 qualifying tuples. HNSW is a graph-based approximate algorithm: search begins at an entry point and traverses the proximity graph using a greedy strategy. With a small `efSearch` (the size of the dynamic candidate list), the search can terminate before all 5 qualifying tuples are visited. The graph topology and the choice of entry point jointly determine which tuples are missed.

Concretely, with a typical HNSW graph constructed over the 10 points and `efSearch = 2`, the search may visit only the 2 closest points to the entry point and return a result set of 2 tuples — say, $(5, 0)$ and $(6, 0)$. The other 3 qualifying tuples — $(3, 0), (4, 0), (7, 0)$ — are not returned, even though they satisfy the radius constraint.

**The failure modes.** HNSW approximation can produce both **false negatives** (qualifying tuples that are not returned) and, less commonly, **false positives** (non-qualifying tuples that are returned due to numerical error in distance computation or graph construction). The logical $\sigma$ has neither. The discrepancy is bounded by HNSW's *recall* — a function of `efSearch`, the construction parameters (`M`, `efConstruction`), and the data distribution.

**Implication for the optimizer.** The logical equivalence rules (A1, V1, J1, etc.) apply to the *logical* $\sigma_{\|V - q\| < \tau}$, not to the physical $H_{V, q, \tau}$. The RBO rewrites logical plans freely; the cost model must account for the fact that the chosen physical implementation may return an *approximation* of the logical result. The brief's separation of logical and physical algebras is what makes this composable.

#### 4.5.2 N2 — Similarity Join Asymmetry Under HNSW

> **N2.** The logical self-join $R \bowtie_{\|V_a - V_b\| < \tau} S$ is symmetric: it equals $S \bowtie_{\|V_b - V_a\| < \tau} R$. The physical implementations are not equivalent in cost: an HNSW index built on $R$ and probed with $S$ has a different cost profile than an HNSW index built on $S$ and probed with $R$.

**Counterexample.** Let $|R| = 10^6$ and $|S| = 10^2$. Suppose both $R$ and $S$ have a vector-typed attribute $V$, and we wish to compute $R \bowtie_{\|V_R - V_S\| < \tau} S$.

*Logical equivalence.* By the commutativity of join and the symmetry of distance, $R \bowtie_{\|V_R - V_S\| < \tau} S = S \bowtie_{\|V_S - V_R\| < \tau} R$ as bags. The two plans produce identical output.

*Physical cost — implementation 1: HNSW on $R$, probe with $S$.* Build an HNSW index over $R$'s vectors: time $O(|R| \cdot \log |R|)$ with construction parameter `M` (typically 16) — roughly $20 \cdot 10^6 = 2 \times 10^7$ operations. Probe with each of the 100 vectors in $S$: $O(|S| \cdot \log |R|) = 100 \cdot 20 = 2000$ operations. Total: approximately $2 \times 10^7$ operations.

*Physical cost — implementation 2: HNSW on $S$, probe with $R$.* Build an HNSW index over $S$: time $O(|S| \cdot \log |S|) = 100 \cdot 7 = 700$ operations. Probe with each of the $10^6$ vectors in $R$: $O(|R| \cdot \log |S|) = 10^6 \cdot 7 = 7 \times 10^6$ operations. Total: approximately $7 \times 10^6$ operations.

The first implementation is approximately **3.5× faster** than the second for this size ratio — and the gap widens for more disparate sizes (e.g., $|R| = 10^7, |S| = 10$ gives a factor of approximately 30×).

**The Volcano exchange pattern.** In Volcano-style execution (Graefe, 1990), a join has a *build* side (materialized into an in-memory structure such as a hash table or an HNSW index) and a *probe* side (consumed tuple-at-a-time). The roles are interchangeable *logically* (the join is commutative), but the cost of building depends on the size of the build side, and the cost of probing depends on the number of probe tuples. The exchange operator can move data between operators, but the build/probe choice is a physical decision the cost model must make.

**Implication for the optimizer.** The optimizer must enumerate *both* physical implementations (HNSW-on-$R$-probe-$S$ and HNSW-on-$S$-probe-$R$) and pick the cheaper one. Join commutativity does not pick a winner; the cost model does. This is the textbook RBO/CBO division, made vivid by the asymmetric costs of vector indices.

#### 4.5.3 N3 — Union Plus Order Does Not Preserve Sort

> **N3.** $\tau_O(R \cup S) \;\not\equiv\; \tau_O(R) \cup \tau_O(S)$ in general. Sorting the union of two bags does not equal the union of two sorted bags.

**Counterexample.** Let $R$ and $S$ be bags over schema $\{\text{id} : \text{int}\}$ with:
- $R = \{(1, 1), (2, 3)\}$, where each tuple is $(\text{id}, \text{score})$; i.e., $R$ contains the tuple `id=1, score=1` and `id=2, score=3`.
- $S = \{(3, 2), (4, 4)\}$: tuples `id=3, score=2` and `id=4, score=4`.

Order by `score` ascending. Consider the rule (which the brief is implicitly testing): *can we sort $R$ and $S$ independently and then union?*

*Left-hand side.* $R \cup S$ under bag semantics contains 4 tuples (each with multiplicity 1):
$$R \cup S = \{(1,1), (2,3), (3,2), (4,4)\}$$
$\tau_{\text{score}}(R \cup S)$ is the sorted list:
$$[\,(1,1),\, (3,2),\, (2,3),\, (4,4)\,]$$
(scores 1, 2, 3, 4 in ascending order). Now take $\Lambda_2$:
$$\Lambda_2(\tau_{\text{score}}(R \cup S)) = [\,(1,1),\, (3,2)\,]$$

*Right-hand side.* $\tau_{\text{score}}(R) = [\,(1,1),\, (2,3)\,]$ and $\tau_{\text{score}}(S) = [\,(3,2),\, (4,4)\,]$. The bag union $\tau_{\text{score}}(R) \cup \tau_{\text{score}}(S)$ contains all 4 tuples (multiplicity 1 each) but is an *unordered* bag — the "sorted" tag of the two inputs is lost in the bag union. The underlying bag is:
$$\{(1,1), (2,3), (3,2), (4,4)\}$$
If we then attempt $\Lambda_2$ on an unordered bag, the result is **non-deterministic**: it could be any 2-tuple sub-bag of the four, e.g., $\{(1,1), (2,3)\}$ or $\{(3,2), (4,4)\}$ or any other 2-tuple selection. It is not necessarily $\{(1,1), (3,2)\}$.

**The non-identity in pure form.** Even setting aside $\Lambda$, the types differ:
- $\tau_O(R \cup S) : \mathcal{O}_S$ (an ordered bag),
- $\tau_O(R) \cup \tau_O(S) : \mathcal{B}_S$ (an unordered bag, the union of two ordered bags).

In a typed reading, the equation is type-incorrect. In an untyped reading (where "equality" means "same underlying bag"), the underlying bags are equal but the order is lost — so a downstream $\Lambda_K$ on the RHS is non-deterministic, whereas the LHS is deterministic.

**Implication for the optimizer.** The rewrite rule "pull $\tau_O$ out of $\cup$" — i.e., $\tau_O(R \cup S) \Rightarrow \tau_O(R) \cup \tau_O(S)$ — is **invalid**. The reverse direction ("push $\tau_O$ into $\cup$") is also invalid. The only safe rewrite is to apply $\tau_O$ to the entire union, after the union is computed. The optimizer must ensure that $\tau_O$ sits above any union, and any $\Lambda_K$ that follows the order must be applied to the *globally sorted* list.

This non-identity is particularly important for multi-source recall queries (e.g., retrieve from multiple memories and merge): the merge step must produce a globally ordered result, not a union of independently ordered sub-results.

---

## Summary

The extended algebra in Section 3 and the equational theory in Section 4 together define the **substrate contract** of MosaicOpt: a BNF grammar and well-formedness preconditions (Section 3.3), the formal bag interpretation of each operator with the four new operators ($\mu$, $\tau$, $\Lambda$, $\Gamma$) defined from first principles (Section 3.4), the foundational axioms (Sections 4.2–4.3), the derived identities with proofs (Section 4.4), and the non-identities with counterexamples (Section 4.5). The non-identities are the most consequential deliverable: they identify the points where the logical algebra and the physical algebra diverge — where HNSW approximation, asymmetric build/probe costs, and the loss of order across set operations break the logical equivalence. These are exactly the points where the brief's three-layer architecture pays off: the RBO reasons about logical plans using the identities, and the CBO reasons about physical cost (including approximation) on the resulting plan space.
