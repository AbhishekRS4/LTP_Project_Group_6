

# Flan-T5
# Single vs few shot

# flan-T5 Flan single shot
# python task_1/test.py \
# --results_save_file final_results/flan_T5_single_shot \
# --model_type google/flan-t5-base \
# --dataset_path datasets/touche23_single_shot_prompt \
# --file_path_model artifacts/T5_base_single_shot_0604-12:44:31/model.ckpt \
# --longT5_mode 0

# # Flan T5 few shot
# python task_1/test.py \
# --results_save_file final_results/flan_T5_few_shot \
# --model_type google/flan-t5-base \
# --dataset_path datasets/touche23_few_shot_prompt \
# --file_path_model artifacts/T5_base_few_shot_0604-15:26:47/f1=0.79.ckpt \
# --longT5_mode 0

# sleep 10
# # ----------------------------------------------------------------------------

# # Flan T5
# # Augmented vs non augmented

# # Flan T5 augmented single shot
# python task_1/test.py \
# --results_save_file final_results/flan_T5_single_shot_augmented \
# --model_type google/flan-t5-base \
# --dataset_path datasets/touche23_single_shot_prompt \
# --file_path_model artifacts/T5_base_augmented_single_shot_0603-14:22:26/model.ckpt \
# --longT5_mode 0

# sleep 10
# # ------------------------------------------------------------------------------

# # Long T5 vs Flan T5
# # Long T5 augmented single shot
# python task_1/test.py \
# --results_save_file final_results/long_T5_single_shot_augmented \
# --model_type google/long-t5-local-base \
# --dataset_path datasets/touche23_long_single_shot_prompt \
# --file_path_model artifacts/longT5_base_augmented_single_shot_0603-18:53:16/model.ckpt \
# --longT5_mode 1


# sleep 10
# ------------------------------------------------------

# Large vs base
# Flan T5 Large augmented single shot
# python task_1/test.py \
# --results_save_file final_results/flan_T5_large_single_shot_augmented \
# --model_type google/flan-t5-base \
# --dataset_path datasets/touche23_single_shot_prompt \
# --file_path_model artifacts/T5_large_augmented_single_shot_0603-22:29:41/model.ckpt \
# --longT5_mode 0 \
# --eval_batch_size 1

# --------------------------------------------------

# best model
#flan-T5 Flan single shot
python task_1/test.py \
--results_save_file final_results/best2_flan_T5_single_shot \
--model_type google/flan-t5-base \
--dataset_path datasets/touche23_single_shot_prompt \
--file_path_model artifacts/T5_base_single_shot_trained_shorter/f1=0.76-v2.ckpt \
--longT5_mode 0