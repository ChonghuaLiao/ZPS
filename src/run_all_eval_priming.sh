
python run_all_eval_with_priming.py \
    --test_split ../config/setting_5/test.list \
    --model_name_or_path [YOUR_MODEL_DIR] \
    --parallelize \
    --template_dir ../templates_test \
    --output_dir ./T0_priming_result
