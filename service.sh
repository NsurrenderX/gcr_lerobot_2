python lerobot/scripts/halo_service.py \
    --policy.type="qwen" \
    --dataset.repo_id="whatever" \
    --dataset.processor='/datassd_1T/qwen25vl/Qwen2.5-VL-3B-Instruct/' \
    --dataset.parent_dir="/data_16T/lerobot_openx/" \
    --data_mix="cup_plus_aug"
    # --uni_res=true \
    # --uni_obs_tensor=true
    
    # --policy.encoder_name="/data_16T/deepseek/xvla_comp/Florence-2-large" \
    # --dataset.processor='/data_16T/deepseek/Qwen2.5-VL-7B-Instruct/' \
    
    # --dataset.root="/data_16T/lerobot_openx/" \
    # --data_mix="pizza"
    # --dataset.parent_dir="/data_16T/lerobot_openx/" \
    # --dataset.root="/data_16T/lerobot_openx/" \
    # --dataset.parent_dir="/data_16T/lerobot_openx/" \
    # --data_mix="pizza_single"
    