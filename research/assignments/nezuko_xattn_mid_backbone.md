# Nezuko assignment marker — Round 21/22

PR: #892 — Mid-backbone surf→vol xattn injection at L=3 + final-layer xattn

Branch: `nezuko/xattn-mid-backbone`
Hypothesis: Insert a second surf→vol cross-attention module after backbone block L=3 (0-indexed, the 3rd of 5 blocks), in addition to the existing final-layer xattn from PR #823 SOTA. This means geometry conditioning from the mid-backbone injection is subsequently processed by backbone blocks L4 and L5, allowing the Transolver to refine the geometry-conditioned representation. Both xattn modules are zero-init (identity at init). Requires adding --xattn-mid-layer CLI flag and splitting the backbone loop in SurfaceTransolver.forward. Surface token positions in the shared sequence are NOT modified by the mid-backbone xattn (volume-only residual update). Directly motivated by PR #887 failure: the failure mode was "random subsampling destroys structural surface coverage" — the right fix is more backbone computation over geometry-conditioned features, not structured selection.

Target: beat val_abupt 6.4407% (single-model SOTA from PR #823). W&B group: nezuko-xattn-mid-backbone.
