cd ..

dataset=mmimdb
dataset_version=top30.json
fusion=GateCls
exp_name=${dataset}_${dataset_version}_clip_freeze_${fusion}
log_dir=/data/zclfe/mm_cls/log/${exp_name}

python main.py \
--log_dir ${log_dir} \
--dataset ${dataset} \
--device 5 \
--logging_steps 10 \
--eval_steps 30 \
--learning_rate 0.005 \
--dataset_version ${dataset_version} \
--fusion ${fusion}