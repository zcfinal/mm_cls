cd ..

exp_name=debug
log_dir=/data/zclfe/mm_cls/log/${exp_name}
dataset=HateMM

python main.py \
--log_dir ${log_dir} \
--dataset ${dataset} \
--device 0