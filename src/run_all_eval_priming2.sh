

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
python run_all_eval_with_priming.py \
    --test_split /home/yanan/shaonan/t-zero/config/setting_5/test_anli_r3.list \
    --model_name_or_path /home/yanan/shaonan/pretrained_model/T0 \
    --parallelize \
    --template_dir /home/yanan/shaonan/t-zero/templates_test \
    --output_dir ./T0_priming_result

#    --template_dir /mfs/shaonan/moonshot/data/temp_dir \
#    --template_dir /mfs/shaonan/moonshot/t-zero/evaluation/ga_t0_norm_shot/ga_configs/step_1 \
