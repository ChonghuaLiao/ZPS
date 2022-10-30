export CUDA_VISIBLE_DEVICES=7
python3 run_model_tuning_self_training_zero_shot.py \
    --test_split ./test_EL.list \
    --model_name_or_path '/localdata/codebook/shaonan/GLM-t0/exp_dir/t0_pretrain_baseline_large_384_32_step10000_nosoftmax/t0-large_nosoftmax/5000' \
    --parallelize \
    --dataset_type pt \
    --ga_dev_distribution ratio \
    --per_device_train_batch_size 4 \
    --num_training_steps 4000 \
    --eval_period 100 \
    --template_dir /localdata/chonghua/templates_test \
    --output_dir ./zl-self_train_32_42
