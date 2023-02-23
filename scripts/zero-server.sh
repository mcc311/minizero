#!/bin/bash
set -e

usage()
{
	echo "Usage: ./zero-server.sh game_type configure_file train_dir end_iteration [OPTION...]"
	echo ""
	echo "  -h, --help                 Give this help list"
	echo "    , --sp_executable_file   Assign the path for self-play executable file"
	echo "    , --op_executable_file   Assign the path for optimization executable file"
	echo "    , --conf_str             Overwrite configuration file"
	exit 1
}

# check argument
if [ $# -lt 4 ]; then
	usage
else
	game_type=$1; shift
	configure_file=$1; shift
	train_dir=$1; shift
	end_iteration=$1; shift
fi

sp_executable_file=build/${game_type}/minizero_${game_type}
op_executable_file=minizero/learner/train.py
overwrite_conf_str=""
while :; do
	case $1 in
		-h|--help) shift; usage
		;;
		--sp_executable_file) shift; sp_executable_file=$1
		;;
		--op_executable_file) shift; op_executable_file=$1
		;;
		--conf_str) shift; overwrite_conf_str=$1
		;;
		"") break
		;;
		*) echo "Unknown argument: $1"; usage
		;;
	esac
	shift
done

run_stage="R"
if [ -d ${train_dir} ]; then
	read -n1 -p "${train_dir} has existed. (R)estart / (C)ontinue / (Q)uit? " run_stage
	echo ""
fi

zero_start_iteration=1
model_file="weight_iter_0.pt"
if [[ ${run_stage,} == "r" ]]; then
	rm -rf ${train_dir}

	echo "create ${train_dir} ..."
	mkdir -p ${train_dir}/model ${train_dir}/sgf
	touch ${train_dir}/op.log
	new_configure_file=$(echo ${train_dir} | awk -F "/" '{ print ($NF==""? $(NF-1): $NF)".cfg"; }')
	cp ${configure_file} ${train_dir}/${new_configure_file}

	# setup initial weight
	echo "\"\" -1 -1" | PYTHONPATH=. python ${op_executable_file} ${game_type} ${train_dir} ${train_dir}/${new_configure_file} >/dev/null 2>&1
elif [[ ${run_stage,} == "c" ]]; then
    zero_start_iteration=$(($(grep -Ei 'optimization.+finished' ${train_dir}/Training.log | wc -l)+1))
    model_file=$(ls ${train_dir}/model/ | grep ".pt$" | sort -V | tail -n1)
    new_configure_file=$(basename ${train_dir}/*.cfg)

	# friendly notification if continuing training
	read -n1 -p "Continue training from iteration: ${zero_start_iteration}, model file: ${model_file}, configuration: ${train_dir}/${new_configure_file}. Sure? (y/n) " yn
	[[ ${yn,,} == "y" ]] || exit
	echo ""
else
	exit
fi

# overwrite configuration file
IFS=':' read -ra settings <<< "${overwrite_conf_str}"
for setting in "${settings[@]}"
do
	IFS='=' read -r key value <<< "${setting}"
	if grep -q "${key}=" ${train_dir}/${new_configure_file}; then
		sed -i "s/^${key}.*/${key}=${value}/g" ${train_dir}/${new_configure_file}
	else
		echo "${key} doesn't exist in ${train_dir}/${new_configure_file}"
		exit
	fi
done

# run zero server
conf_str="zero_training_directory=${train_dir}:zero_end_iteration=${end_iteration}:nn_file_name=${model_file}:zero_start_iteration=${zero_start_iteration}"
${sp_executable_file} -conf_file ${train_dir}/${new_configure_file} -conf_str ${conf_str} -mode zero_server
