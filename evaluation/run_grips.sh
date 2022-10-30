

export CUDA_VISIBLE_DEVICES=1,2,3,4
python run_grips.py \
    --model_name_or_path /home/yanan/shaonan/pretrained_model/T0 \
    --parallelize \
    --ga_dev_distribution ratio \
    --template_dir /home/yanan/shaonan/t-zero/evaluation/templates_test \
    --output_dir ./grips_template

#    --template_dir /mfs/shaonan/moonshot/data/temp_dir \
#    --template_dir /mfs/shaonan/moonshot/t-zero/evaluation/ga_t0_norm_shot/ga_configs/step_1 \
