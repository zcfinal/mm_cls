cd ..

dataset=mmimdb
exp_name=${dataset}_clip_freeze_mlp
log_dir=/data/zclfe/mm_cls/log/${exp_name}

python main.py \
--log_dir ${log_dir} \
--dataset ${dataset} \
--device 5 \
--logging_steps 10 \
--eval_steps 100 \
--learning_rate 0.005