

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python run_model_tuning.py \
    --test_split /home/yanan/shaonan/t-zero/config/setting_5/test.list \
    --model_name_or_path /home/yanan/shaonan/pretrained_model/T0 \
    --parallelize \
    --dataset_type pt \
    --ga_dev_distribution ratio \
    --per_device_train_batch_size 4 \
    --num_training_steps 4000 \
    --eval_period 100 \
    --template_dir /home/yanan/shaonan/t-zero/templates_test \
    --output_dir ./original_mt_with_prompt