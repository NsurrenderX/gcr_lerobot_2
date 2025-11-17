python lerobot/scripts/pi0_service.py \
    --policy.type="pi0" \
    --dataset.repo_id="whatever" \
    --dataset.processor='/data_16T/deepseek/Qwen2.5-VL-7B-Instruct/' \
    --dataset.root="/data_16T/lerobot_openx/" \
    --data_mix="cup_pp_50" \
    --uni_res=true \
    --uni_obs_tensor=true
    
    # --policy.encoder_name="/data_16T/deepseek/xvla_comp/Florence-2-large" \
    # --dataset.processor='/data_16T/deepseek/Qwen2.5-VL-7B-Instruct/' \
    
    # --dataset.root="/data_16T/lerobot_openx/" \
    # --data_mix="pizza"
    # --dataset.parent_dir="/data_16T/lerobot_openx/" \
    # --dataset.parent_dir="/data_16T/lerobot_openx/" \
    # --data_mix="pizza_single"
    