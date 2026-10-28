# Evidence-constrained auditing

An audit is a detector and evidence collector, not an editor. Keep extraction, audit, and writeback as
separate trust domains. A model's confidence or mathematical plausibility never makes a reading source true.

## Required evidence protocol

For transcription and formula conflicts:

1. Build a deterministic checklist from existing candidates before any visual call. Checklist atoms are
   numbers, variables, subscripts, exponents, signs, relations, and exhaustive table/list cells.
2. Run three fresh candidate-blind visual reads. Do not show candidate statements, candidate values, prior
   audit conclusions, or a candidate-derived math template. Tell every observer that the printed book may be
   inconsistent or wrong and must be copied literally.
3. Require one unique 2-of-3 normalized reading for every checklist atom. Save all attempts, including low
   confidence, parse failures, and dissenting values.
4. Match the consensus deterministically against the existing candidate atom maps. Automatic writeback is
   allowed only when exactly one existing candidate matches every atom.
5. Reuse that candidate statement byte-for-byte. Never apply a model-generated `verified_statement`.
6. If the consensus contains any value absent from all existing candidates, record
   `novel_observation_requires_human_verification`. This remains unresolved even when all three observers agree.

The last rule prevents correlated semantic repair. For example, a printed formula may use `n` while nearby
prose quantifies `i`. Multiple capable models can independently replace the visible formula with a more natural
`i`-indexed expression. Agreement does not prove pixel fidelity when every observer shares the same prior.

### Source-glyph hard gate

Any finding that would change an equality, congruence sign, negation, inequality, quantifier, matrix entry,
subscript, exponent, summation bound, or variable name must pass an additional glyph-level gate:

1. render the full physical page first, then a 4x or 5x crop that includes the disputed glyph and enough nearby
   text to identify the object;
2. require the reviewer to report the exact visible glyph sequence and crop coordinates before giving any
   mathematical interpretation;
3. reject a proposed correction when it merely makes the mathematics more conventional, internally consistent,
   or true but the pixels do not support it;
4. when the pixels visibly contain an awkward or apparently false expression, preserve the literal expression and
   attach a `source_anomaly` note rather than silently normalizing it;
5. if the full page and the crop disagree because the original bbox was wrong, discard the bbox result and restart
   from the full page. Never delete the canonical object solely because a crop was empty or mislocalized.

Independent adjudicators must inspect the rendered evidence themselves. A second model that only reads another
reviewer's prose is not independent visual verification. Include deliberately awkward literal expressions as
negative controls in each recurring regression stratum; a reviewer that "fixes" them fails the audit batch.

For canonical role, duplicate, split, merge, and context-only decisions:

- use three independent visual decisions and require an exact 2-of-3 action consensus;
- compare `action`, selected candidate, and context-only IDs deterministically;
- for `retain_distinct`, candidate selection is irrelevant;
- for `merge`, keep the selected existing candidate text unchanged;
- never use a role audit to invent missing statement text or repair notation.

## Finding states

Use explicit states rather than a single `confirmed` flag:

- `suspected`: anomaly found by heuristics or a free-form reviewer;
- `visually_observed`: one or more literal reads exist, but the writeback gate is not satisfied;
- `confirmed_existing_candidate`: blind consensus uniquely selected an existing candidate;
- `novel_observation_requires_human_verification`: observers propose a value outside the candidate set;
- `human_pdf_visual_verification`: a person checked the cited physical page and supplied a narrow override;
- `rejected`: evidence contradicted the finding or the claimed source span was wrong.

Free-form blind reviews are useful for discovering candidate cases, but their assertions stay `suspected` until
the bounded protocol above succeeds. Do not tune extraction prompts to every suspected case. First reproduce
the issue against the physical PDF, verify the bbox/page, and run at least one negative control where the
auditor should leave a deliberately awkward but literal source expression unchanged.

For boundary and readability findings, audit the complete consumer-facing record, not `statement` in isolation.
Read `context_before`, `source_context`, their host IDs, and the source-context group. An anaphoric statement with
an exact, high-confidence context link is a valid atomic-plus-context representation; report a defect only when
the link is absent, wrong, or still insufficient. Conversely, a complete parent stored only as provenance does
not excuse multiple canonical children that repeat the same source claim. This distinction keeps the auditor
from oscillating between over-merging and over-splitting on successive rounds.

## Reporting

Report both true and false audit findings. Include the sample denominator, finding state, page/bbox, literal
critical atoms, observer disagreement, whether the value existed in a candidate, and whether any writeback
occurred. Never describe detector coverage, model agreement, or `uncertain_spans = 0` as human-gold accuracy.
