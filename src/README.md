# Reproducing main results

Here we provide some details to reproduce the main numbers we reported in the paper using PyTorch and 🤗 Transformers. The original T0 prompts from [Prompt Source](https://github.com/bigscience-workshop/promptsource) can be found in `../templates_test`. Before using the scripts, you should replace `[YOUR_MODEL_DIR]` in each script with the path to your model.

## Zero-label
To launch ZPS evaluation or self-training, directly use `bash run_zps.sh` or `bash run_self_trining.sh`.

## Few-shot
- GRIPS: We provide prompts searched with grips in `../grips_template`, you can just evaluate T0-11B with this directory. e.g. 
    ```bash
    python run_zps.py \
        --test_split ../config/setting_5/test.list \
        --model_name_or_path [YOUR_MODEL_DIR] \
        --template_dir ../grips_template \
        --dataset_type all \
        --ga_dev_distribution ratio \
        --parallelize \
        --output_dir results

    ```
    or you can use `bash run_grips.sh` to run the grips algorithm.
- ICL: `bash run_all_eval_priming.sh`
- GPS: Use `python run_gps.py` to run GPS. You should change `[YOUR_MODEL_PATH]` and `[YOUR_T5XXL_PATH]` to the path to T0 and T5-xxl respectively.

