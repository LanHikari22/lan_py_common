#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
project_dir="$script_dir"

# Used this to install a specific mypy because of bug https://github.com/dry-python/classes/issues/481
#python3 -m pipx install 'mypy==0.971' --suffix=@0.971

#python3 -m pipx inject --quiet mypy@0.971 classes
#~/.local/share/pipx/venvs/mypy@0-971/bin/python3 -m pip install -e $project_dir

#PYTHONPATH=$project_dir/src/$(cat $project_dir/app_name) python3 -m $(cat $project_dir/app_name).test $@
#PYTHONPATH=$project_dir/src/$(cat $project_dir/app_name) script -q -c "mypy@0.971 --strict -m "$(cat $project_dir/app_name)".test" /dev/null | tac
#mypy@0.971 --strict -m lan_py_common.test

#mypy --strict -m lan_py_common.test
script -q -c "mypy --strict -m lan_py_common" /dev/null | tac

#script -q -c "cd src; mypy --strict -m newton_method.main" /dev/null | tac
