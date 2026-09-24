# Python (tempfile) and uv write scratch files and locks to TMPDIR. Point it at
# R's session temp directory, which is removed when the test process exits, so
# R CMD check does not report detritus in the temp directory.
withr::local_envvar(TMPDIR = tempdir(), .local_envir = testthat::teardown_env())
