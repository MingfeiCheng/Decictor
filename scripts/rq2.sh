#!/bin/bash

project_root=/data/d/mingfeicheng/Decictor
method_name_list='f_behavior f_oracle f_path wo_constraint wo_motion wo_remove'
repeat_lst="1 2 3 4 5 6 7 8 9 10"
run_hour=4

declare -a seed_infos=(
    "scenario_1 sunnyvale_loop"
    "scenario_2 sunnyvale_loop"
    "scenario_3 sunnyvale_loop"
    "scenario_4 sunnyvale_loop"
    "scenario_5 sunnyvale_big_loop"
    "scenario_6 sunnyvale_big_loop"
)

run_test() {
    run_id=$1
    seed_name=$2
    map_name=$3
    method_name=$4

    run_name=run_$run_id

    python ${project_root}/main_fuzzer.py \
      fuzzer=$method_name \
      fuzzer.run_hour=$run_hour \
      seed_name=$seed_name \
      map_name=$map_name \
      run_name="$run_local_name"
}

for run_id in $repeat_lst; do
  for seed_info in "${seed_infos[@]}"; do
    for method_name in $method_name_list; do
      IFS=' ' read -ra seed_details <<< "$seed_info"
      run_test "${run_id}" "${seed_details[0]}" "${seed_details[1]}" "${method_name}"
    done
  done
done