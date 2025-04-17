#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
project_dir="$script_dir"

PYTHONPATH=$project_dir/src/$(cat $project_dir/app_name) python3 -m $(cat $project_dir/app_name).lib $@
