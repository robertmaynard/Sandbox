

## benchmark_runner ##
A simple python script that runs all executables/benchmarks from a build
directory and saves the output to log files


## benchmark_uploader ##
Given a pattern and a directory uploads all files that match to a given
s3 bucket


## benchmark_converter ##
A simple python script that converts the VTK-m benchmark output to json format.

The format of the json is in the template.json file

This requires that the python script is run from a build directory, so that
it can determine the "vcs" information including for git: "sha", "url", and "branch".



