## Test environments

* Local Ubuntu 24.04, R 4.3.3, `R CMD check --as-cran` (NOT_CRAN=true so the test suite runs)
* GitHub Actions R-CMD-check: macOS release, Windows release, Ubuntu devel, Ubuntu release, Ubuntu oldrel-1, all with `--as-cran`

## R CMD check results

0 errors, 0 warnings, 0 notes on the GitHub Actions `--as-cran` runs (those runs do not repeat the incoming URL fetch). The citation URL is `https://doi.org/10.5117/CCR2026.2.2.TOMA`. CRAN checks that DOI through the handle API, which returns 200, rather than the publisher page.

Installed package size is about 3.7 MB (was about 4.6 MB in 0.1.7), under the 5 MB threshold. The source tarball is about 3.0 MB.

## This release

Version 0.1.8. The Title and Description now cover speech, face finding, and the local coding task, in addition to text, image, and video. `googledrive`, `httpuv`, and `pbapply` were already in Imports and are now imported in the NAMESPACE. `LSAfun` and `remotes` are suggested and loaded only when `nlp_scores()` and `emoxicon_scores()` need them.

`emotions.rda` and `tinytrolls.rda` were already saved with xz compression; resaving did not shrink them, and the objects are unchanged. The README logo is no longer installed (no help page uses it). Two example PNGs were recompressed without changing the pixels.
