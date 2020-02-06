Proof of concept that allows CTest to only run the subset of tests whose
dependencies has changed since the last CTest run. Goal is to allow people
to automatically run just the tests that are impacted by a set of changes.
Good for CI / incremental builds.

Requires the git format patch to be applied to CMake to work 'properly'
This is still very much a proof of concept
