#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
project_dir="$script_dir"

cd $project_dir && pytest
