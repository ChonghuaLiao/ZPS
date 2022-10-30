CUDA_VISIBLE_DEVICES=3 python use_dino_to_generate_template.py  \
                        --task_list_file /mfs/shaonan/moonshot/t-zero/config/setting_5/test_temp.list \
                        --input_dir /mfs/shaonan/moonshot/t-zero/templates \
                        --output_dir /mfs/shaonan/moonshot/data/temp_dir \

                        # --input_dir /mfs/shaonan/moonshot/t-zero/templates
                        # --model_name /mfs/shaonan/pretrained_model/t5-xxl-lm-adapt