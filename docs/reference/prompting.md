# Prompt adaptation patterns

## Table of contents

1. Inventory/segmentation overlays
2. Visual overlays
3. Distant support
4. Worked examples
5. Coverage traps
6. Problem compendia and survey articles

## 1. Inventory/segmentation overlays

Add observable rules, for example:

- “Exercises under an `Exercises` heading begin with bare labels such as `B.10`; do not require the
  word Exercise.”
- “Difficulty glyphs immediately following the exercise number belong to the statement blocks.”
- “A boxed Algorithm caption and its pseudocode form one object.”

Never ask inventory or MinerU segmentation to correct formulas or reproduce text. Its output is
labels, boundaries, short cues, or IDs only.

## 2. Visual overlays

Use overlays for source typography that changes crop interpretation:

- theorem statement and proof share one colored box;
- example resolution begins after a printed “Solution” token in the same block;
- page-side annotations and footnotes must be excluded;
- multi-column reading order differs from MinerU order.

Never add mathematical facts not visible in the crop.

## 3. Distant support

Answer sections often appear hundreds of pages later. Segment them as records with empty
`statement_ids`, matching kind/label, and populated `supporting_ids`. Global merge attaches the
longest source-supported candidate. If labels restart by chapter, require full chapter labels before
enabling this merge.

## 4. Worked examples

- A visible question/setup followed by reasoning/result: statement + supporting text.
- A short declarative example with no separable resolution: all in statement.
- An example and exercise with the same number: distinct `(kind,label)` records.
- Never rewrite an example into an artificial question merely to fill both Q/A fields.

## 5. Coverage traps

Watch for appendix labels, split/merged adjacent labels, first items beginning mid-section,
continuations crossing page/chunk boundaries, contents/index false positives, equation numbers,
proofs captured as separate theorems, footnotes inside crops, and referenced figures omitted from
the asset list.

## 6. Problem compendia and survey articles

Some books label long survey articles as “Problem A1”, “Problem D18”, and so on. The article can contain an
open question plus definitions, known theorems, bounds, constructions, counterexamples, numerical examples,
and bibliography over several pages. Treat the article heading as `record_type=section`; extract its atomic
mathematical units separately. Do not create a giant exercise containing “the minimum setup” because that
silently drops later results and duplicates any children that are also extracted.

Continuation pages are high risk. The independent coverage audit must scan them even if the article's main
question was already captured. For source-faithful self-containedness, extend the literal source span or copy
the minimum visibly shared antecedent into each canonical statement. `context_before`/`context_after`,
`source_context`, and dependency edges may additionally preserve provenance, but they do not make the exported
QA question self-contained. Do not rewrite pronouns from memory or silently add mathematical premises.
