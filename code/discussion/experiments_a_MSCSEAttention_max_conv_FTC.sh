#!/bin/bash

# nohup bash experiments_a_MSCSEAttention_max_conv_FTC.sh >> experiments_a_MSCSEAttention_max_conv_FTC.log 2>&1 &

# 固定参数部分（保持不变）
BASE_CMD="python ./IncluRCA/trainer/a_rca_trainer.py \
--window_size 11 \
--data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
--dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
--epochs 300 \
--orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 \
--efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 \
--eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 \
--GAT_name3 GATv2Conv --GAT_name4 GATConv --GAT_name5 GATConv \
--activ_fun3 relu6 --activ_fun4 relu6 --activ_fun5 relu6"

# 可变参数
squeeze_types=("max")
excite_types=("conv")
GAT_NAMES=("GATConv" "GATv2Conv")
ACTIV_FUNS=("silu" "rrelu" "leaky_relu" "celu" "selu" "elu" "relu6")

# 遍历所有组合
for sq in "${squeeze_types[@]}"; do
  for ex in "${excite_types[@]}"; do
    for gat1 in "${GAT_NAMES[@]}"; do
      for gat2 in "${GAT_NAMES[@]}"; do
        for act1 in "${ACTIV_FUNS[@]}"; do
          for act2 in "${ACTIV_FUNS[@]}"; do

            # 构造唯一标识符（用于路径和日志名）
            ID="${sq}_${ex}_${gat1}_${gat2}_${act1}_${act2}"
            MODEL_DIR="/root/shared-nvme/work/code/RCA/IncluRCA/model/a3/checkpoint/${ID}"
            MODEL_PATH="$MODEL_DIR/main.pt"
            LOG_FILE="/root/shared-nvme/work/code/RCA/IncluRCA/code/logs/train_${ID}.log"

            # 如果模型或日志已存在，跳过
            if [[ -f "$MODEL_PATH" ]] || [[ -f "$LOG_FILE" ]]; then
              echo "⚠️ Skipping already completed run: $ID"
              continue
            fi

            echo "============================================"
            echo "MultiScaleConvSEAttention reduction=32"
            echo "Running: $ID"
            echo "============================================"

            mkdir -p "$MODEL_DIR"

            # 执行训练命令
            $BASE_CMD \
              --model_path "$MODEL_PATH" \
              --squeeze_type "$sq" \
              --excite_type "$ex" \
              --GAT_name1 "$gat1" \
              --GAT_name2 "$gat2" \
              --activ_fun1 "$act1" \
              --activ_fun2 "$act2" \
              | tee "$LOG_FILE"

            echo "✅ Finished: $ID → log saved to $LOG_FILE"
            echo ""

          done
        done
      done
    done
  done
done

echo "🎉 All new combinations completed!"