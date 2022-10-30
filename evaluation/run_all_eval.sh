

export CUDA_VISIBLE_DEVICES=1,2
python run_all_eval.py \
    --test_split /mfs/shaonan/moonshot/t-zero/config/setting_5/test_temp.list \
    --model_name_or_path /mfs/shaonan/pretrained_model/T0_3B \
    --parallelize \
    --dataset_type ga \
    --template_dir /mfs/shaonan/moonshot/t-zero/evaluation/ga_t0_norm_shot/ga_configs/step_1 \
    --output_dir ./debug \
    --debug

#    --template_dir /mfs/shaonan/moonshot/data/temp_dir \
#    --template_dir /mfs/shaonan/moonshot/t-zero/evaluation/ga_t0_norm_shot/ga_configs/step_1 \

num=0
while (( $num < 7 ))
do
    python /home/yanan/chonghua/GPS_clean/evaluation/run_all_eval_leave_one_out.py \
        --test_split /home/yanan/chonghua/GPS_clean/config/setting_5/test_bak.list \
        --model_name_or_path /home/yanan/shaonan/pretrained_model/T0 \
        --template_dir /home/yanan/chonghua/GPS_clean/templates_test \
        --dataset_type all \
        --ga_dev_distribution ratio \
        --parallelize \
        --output_dir /home/yanan/chonghua/GPS_clean/evaluation/results
done