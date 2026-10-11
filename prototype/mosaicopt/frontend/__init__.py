"""SQL/MM-style frontend that desugars SQL to the algebraic plan.

The frontend accepts a restricted subset of SQL (extended with the
MosaicOpt predicates) and emits a logical plan tree.

Supported surface (the Q1->Q6 example):
    WITH scored AS (
      SELECT id, text, ts, subsystem,
             (emb <=> :q1) AS d1,
             (emb_alt <=> :q2) AS d2
      FROM memory
      WHERE subsystem = 'fault-tolerance'
        AND ts > :recent
    )
    SELECT id, text, ts
    FROM scored
    WHERE d1 < :tau1 AND d2 < :tau2
    ORDER BY 0.7 * d1 + 0.3 * d2
    LIMIT 10;
"""

from .sql import parse_sql, parse_to_plan

__all__ = ["parse_sql", "parse_to_plan"]
