python run_zps.py \
    --test_split ../config/setting_5/test.list \
    --model_name_or_path [MODEL_PATH] \
    --template_dir ../templates_test \
    --dataset_type all \
    --ga_dev_distribution ratio \
    --parallelize \
    --output_dir results
