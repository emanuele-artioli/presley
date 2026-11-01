# PRESLEY numerical figure artifact

This package preserves the numerical inputs for eight manuscript figures:

| Figure | Input | Generator |
|---|---|---|
| 4: corpus breadth | `figures/breadth.json` | `tools/plot_breadth.py` |
| 5: matched-rate comparison with ELVIS | `figures/ratematched.json` | `tools/plot_ratematched.py` |
| 6: regime reversal | `figures/regime_reversal.json` | `tools/plot_regime_reversal.py --plot-data ...` |
| 7: restorer catalogue | `figures/restorer_catalogue.json` | `tools/plot_restorer_catalogue.py` |
| 9: stage timing | `figures/stage_timing.json` | `tools/plot_stage_timing.py` |
| 11 left: selection cost | `figures/selection_cost.json` | `tools/plot_selection_cost.py` |
| 11 right: restorability | `figures/restorability_plot.json` | `tools/plot_restorability.py --plot-data ...` |
| 12: resolution ladder | `figures/resolution_ladder.json` and `.desc.tex` | `tools/plot_resolution_ladder.py` |

Run `python tools/reproduce_paper_figures.py --out <new-directory>` from the
repository root after installing `requirements-figures.txt`. The wrapper stages
the historical directory layout in temporary storage; it never writes into an
Overleaf checkout or the original result tree. It verifies `manifest.json`
checksums first and records generator output in `reproduction.log`.

The JSON inputs were restored byte-for-byte from paper commit
`68153d43f6717e7cf362c4cd3bc29d82f505b8dc`, immediately before auxiliary files
were removed for submission. The manifest records original paths, Git objects
and SHA-256 digests. The reference PDFs are the existing submitted figures from
paper revision `de616f9678f488c1e2367bde1417a720e580daa7`.

For restorability, the extra plot-data export contains the original deterministic
scatter subsample, 120 within-run correlations, and the regression evaluated
on the original full set of points. Its provenance records the raw NPZ digest
and the plotting source digest. No model, encode, restoration or metric backfill
is run to create this export. The generator checks the published run count,
positive count and median correlation. The regime-reversal generator recomputes
the four BD-rates and checks them against the manuscript.

Redrawing archived numbers is distinct from recomputing measurements from source
videos. The latter requires the original datasets, masks, checkpoints, result
records and EVCA caches, which are not distributed in this small artifact.
The qualitative image and the pipeline diagrams are also outside this numerical
redraw workflow. `qualitative.json` preserves the original illustrative crop's
selection metadata; it is not additional quantitative evidence. Fonts, PDF
metadata and layout can vary by plotting environment; the archived reference
PDFs remain unchanged.
