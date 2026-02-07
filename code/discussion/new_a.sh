# # #！/bin/bash
# # # nohup bash new_a.sh >> new_a.log 2>&1 &

# LOG_FILE="/root/shared-nvme/work/code/RCA/IncluRCA/code/new_a.log"

# # Step 1: 从日志中提取所有已完成的 (gat1, gat2, act1, act2) 组合，并存入关联数组（Bash 4+）
# declare -A DONE_SET

# if [[ -f "$LOG_FILE" ]]; then
#     awk '
#     /GAT_name1:/ { g1 = $2 }
#     /GAT_name2:/ { g2 = $2 }
#     /activ_fun1:/ { a1 = $2 }
#     /activ_fun2:/ { 
#         a2 = $2
#         if (g1 != "" && g2 != "" && a1 != "" && a2 != "") {
#             print g1 "|" g2 "|" a1 "|" a2
#         }
#         g1 = g2 = a1 = a2 = ""
#     }
#     ' "$LOG_FILE" > /tmp/.done_keys.$$

#     while IFS= read -r key; do
#         [[ -n "$key" ]] && DONE_SET["$key"]=1
#     done < /tmp/.done_keys.$$
#     rm -f /tmp/.done_keys.$$
# else
#     echo "Log file not found: $LOG_FILE, assuming no experiments completed."
# fi

# # 调试：打印所有已记录的已完成组合
# if [[ ${#DONE_SET[@]} -gt 0 ]]; then
#     echo "=== 已检测到以下已完成的实验组合（共 ${#DONE_SET[@]} 个）==="
#     for key in "${!DONE_SET[@]}"; do
#         echo "  $key"
#     done
# else
#     echo "=== 未发现任何已完成的实验组合 ==="
# fi

# echo "---Main Experiment---"
# echo "---Resuming with skip of already-run combinations---"

# GAT_NAMES=("GATConv" "GATv2Conv")
# ACTIV_FUNS=("hardtanh" "mish" "hardswish" "silu" "tanh" "rrelu" "leaky_relu" "celu" "selu" "elu" "relu6")

# BASE_CMD="python ./IncluRCA/trainer/a_rca_trainer.py \
#   --window_size 11 \
#   --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
#   --dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
#   --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a1/checkpoint/main.pt \
#   --epochs 300 \
#   --orl_te_heads 2 \
#   --orl_te_layers 2 \
#   --orl_te_in_channels 256 \
#   --efi_in_dim 256 \
#   --efi_te_heads 4 \
#   --efi_te_layers 2 \
#   --efi_out_dim 256 \
#   --eff_in_dim 256 \
#   --eff_GAT_out_channels 128 \
#   --eff_GAT_heads 2 \
#   --batch_size 64 \
#   --eff_GAT_dropout 0.1 \
#   --explainer_mask_acti_func relu \
#   --GAT_name3 GATv2Conv \
#   --GAT_name4 GATConv \
#   --GAT_name5 GATConv \
#   --activ_fun3 relu6 \
#   --activ_fun4 relu6 \
#   --activ_fun5 relu6"

# total=0
# skipped=0
# run_count=0

# for gat1 in "${GAT_NAMES[@]}"; do
#   for gat2 in "${GAT_NAMES[@]}"; do
#     for act1 in "${ACTIV_FUNS[@]}"; do
#       for act2 in "${ACTIV_FUNS[@]}"; do
#         total=$((total + 1))
#         key="$gat1|$gat2|$act1|$act2"

#         if [[ -n "${DONE_SET[$key]}" ]]; then
#             echo "[SKIP] Already done: GAT1=$gat1, GAT2=$gat2, ACT1=$act1, ACT2=$act2"
#             skipped=$((skipped + 1))
#             continue
#         fi

#         CMD="$BASE_CMD \
#           --GAT_name1 $gat1 \
#           --GAT_name2 $gat2 \
#           --activ_fun1 $act1 \
#           --activ_fun2 $act2"

#         echo "begin----------------------------------------"
#         echo "Running ($((run_count + 1))): GAT1=$gat1, GAT2=$gat2, ACT1=$act1, ACT2=$act2"
#         echo "Command: $CMD"

#         # 执行命令
#         eval "$CMD"
#         run_count=$((run_count + 1))

#         # 可选：每完成一个就追加标记到日志（防止中断后重复）
#         # 但原程序已在日志输出关键字段，所以可不加

#       done
#     done
#   done
# done

# echo "--------------------------------------------------"
# echo "Total combinations: $total"
# echo "Skipped (already done): $skipped"
# echo "Actually executed this run: $run_count"
# echo "Done."

#!/bin/bash

LOG_FILE="/root/shared-nvme/work/code/RCA/IncluRCA/code/new_a.log"
JSON_FILE="/root/shared-nvme/work/code/RCA/IncluRCA/code/experiments_GATNet_FTC/good_configs.json"

# Step 1: 从日志中提取所有已完成的 (gat1, gat2, act1, act2) 组合，并存入关联数组
declare -A DONE_SET

if [[ -f "$LOG_FILE" ]]; then
    awk '
    /GAT_name1:/ { g1 = $2 }
    /GAT_name2:/ { g2 = $2 }
    /activ_fun1:/ { a1 = $2 }
    /activ_fun2:/ { 
        a2 = $2
        if (g1 != "" && g2 != "" && a1 != "" && a2 != "") {
            print g1 "|" g2 "|" a1 "|" a2
        }
        g1 = g2 = a1 = a2 = ""
    }
    ' "$LOG_FILE" > /tmp/.done_keys.$$

    while IFS= read -r key; do
        [[ -n "$key" ]] && DONE_SET["$key"]=1
    done < /tmp/.done_keys.$$
    rm -f /tmp/.done_keys.$$
else
    echo "Log file not found: $LOG_FILE, assuming no experiments completed."
fi

# 调试：打印所有已记录的已完成组合
if [[ ${#DONE_SET[@]} -gt 0 ]]; then
    echo "=== 已检测到以下已完成的实验组合（共 ${#DONE_SET[@]} 个）==="
    for key in "${!DONE_SET[@]}"; do
        echo "  $key"
    done
else
    echo "=== 未发现任何已完成的实验组合 ==="
fi

# Step 2: 检查 JSON 文件是否存在
if [[ ! -f "$JSON_FILE" ]]; then
    echo "ERROR: JSON 配置文件不存在: $JSON_FILE"
    exit 1
fi

echo "---Main Experiment---"
echo "---Running only the good configurations from JSON---"

BASE_CMD="python ./IncluRCA/trainer/a_rca_trainer.py \
  --window_size 11 \
  --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
  --dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
  --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a1/checkpoint/main.pt \
  --epochs 300 \
  --orl_te_heads 2 \
  --orl_te_layers 2 \
  --orl_te_in_channels 256 \
  --efi_in_dim 256 \
  --efi_te_heads 4 \
  --efi_te_layers 2 \
  --efi_out_dim 256 \
  --eff_in_dim 256 \
  --eff_GAT_out_channels 128 \
  --eff_GAT_heads 2 \
  --batch_size 64 \
  --eff_GAT_dropout 0.1 \
  --explainer_mask_acti_func relu \
  --GAT_name3 GATv2Conv \
  --GAT_name4 GATConv \
  --GAT_name5 GATConv \
  --activ_fun3 relu6 \
  --activ_fun4 relu6 \
  --activ_fun5 relu6"

total=0
skipped=0
run_count=0

# 使用 jq 从 JSON 中提取配置并遍历
while IFS= read -r line; do
    # line 格式: GAT1|GAT2|ACT1|ACT2
    gat1=$(echo "$line" | cut -d'|' -f1)
    gat2=$(echo "$line" | cut -d'|' -f2)
    act1=$(echo "$line" | cut -d'|' -f3)
    act2=$(echo "$line" | cut -d'|' -f4)

    total=$((total + 1))
    key="$gat1|$gat2|$act1|$act2"

    if [[ -n "${DONE_SET[$key]}" ]]; then
        echo "[SKIP] Already done: GAT1=$gat1, GAT2=$gat2, ACT1=$act1, ACT2=$act2"
        skipped=$((skipped + 1))
        continue
    fi

    CMD="$BASE_CMD \
      --GAT_name1 $gat1 \
      --GAT_name2 $gat2 \
      --activ_fun1 $act1 \
      --activ_fun2 $act2"

    echo "begin----------------------------------------"
    echo "Running ($((run_count + 1))): GAT1=$gat1, GAT2=$gat2, ACT1=$act1, ACT2=$act2"
    echo "Command: $CMD"

    eval "$CMD"
    run_count=$((run_count + 1))

done < <(jq -r '.[] | "\(.config.GAT_name1)|\(.config.GAT_name2)|\(.config.activ_fun1)|\(.config.activ_fun2)"' "$JSON_FILE")

echo "--------------------------------------------------"
echo "Total good configurations from JSON: $total"
echo "Skipped (already done): $skipped"
echo "Actually executed this run: $run_count"
echo "Done."