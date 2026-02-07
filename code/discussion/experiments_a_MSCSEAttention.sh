#！/bin/bash
# nohup bash experiments_a_MSCSEAttention.sh >> experiments_a_MSCSEAttention.log 2>&1 &

# 固定参数部分（保持不变）
BASE_CMD="python ./IncluRCA/trainer/a_rca_trainer.py \
--window_size 11 \
--data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
--dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
--epochs 300 \
--orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 \
--efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 \
--eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 \
--GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu \
--GAT_name3 GATv2Conv --GAT_name4 GATConv --GAT_name5 GATConv \
--activ_fun3 relu6 --activ_fun4 relu6 --activ_fun5 relu6"

# 可变参数
squeeze_types=("avg" "max" "avg_max" "conv")
excite_types=("fc" "conv" "multi_conv")

# 遍历所有组合
for sq in "${squeeze_types[@]}"; do
  for ex in "${excite_types[@]}"; do
    # 构造模型保存路径
    MODEL_DIR="/root/shared-nvme/work/code/RCA/IncluRCA/model/a3/checkpoint/${sq}_${ex}"
    MODEL_PATH="$MODEL_DIR/main.pt"
    LOG_FILE="train_${sq}_${ex}.log"

    # 如果模型文件或日志文件已存在，跳过该组合
    if [[ -f "$MODEL_PATH" ]] || [[ -f "$LOG_FILE" ]]; then
      echo "⚠️ Skipping already completed run: squeeze_type=$sq, excite_type=$ex"
      continue
    fi

    echo "============================================"
    echo "Running: squeeze_type=$sq, excite_type=$ex"
    echo "============================================"

    # 创建模型目录
    mkdir -p "$MODEL_DIR"

    # 执行训练命令，并同时输出到终端和日志文件
    $BASE_CMD \
      --model_path "$MODEL_PATH" \
      --squeeze_type "$sq" \
      --excite_type "$ex" \
      | tee "$LOG_FILE"

    echo "✅ Finished: $sq + $ex → log saved to $LOG_FILE"
    echo ""
  done
done

echo "🎉 All new combinations completed!"