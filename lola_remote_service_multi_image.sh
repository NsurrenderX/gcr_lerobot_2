python lerobot/scripts/inference_server_lola_multi_image.py \
    --policy.type="qwen" \
    --policy.qwen_path="../Qwen2.5-VL-3B-Instruct/" \
    --dataset.repo_id="whatever" \
    --dataset.processor='../Qwen2.5-VL-3B-Instruct/' \
    --dataset.parent_dir="../" \
    --data_mix="aloha_bb_extended" \

#--policy.qwen_path="../Qwen2.5-VL-7B-Instruct/" \
#--dataset.repo_id="whatever" \
#--dataset.processor="../Qwen2.5-VL-7B-Instruct/" \
