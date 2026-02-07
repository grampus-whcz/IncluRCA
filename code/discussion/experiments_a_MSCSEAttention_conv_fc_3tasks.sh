#!/bin/bash
# nohup bash experiments_a_MSCSEAttention_conv_fc_3tasks.sh >> experiments_a_MSCSEAttention_conv_fc_3tasks.log 2>&1 &

set -e

BASE_TRAIN_CMD="python ./IncluRCA/trainer/a_rca_trainer.py \
--window_size 11 \
--data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
--dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
--epochs 300 \
--orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 \
--efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 \
--eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 \
--squeeze_type conv \
--excite_type fc"

BASE_LOCALIZER_CMD="python ./IncluRCA/trainer/a_localizer.py \
--window_size 11 \
--data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
--dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
--epochs 300 \
--orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 \
--efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 \
--eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 \
--explainer_mask_acti_func relu \
--GAT_name3 GATv2Conv --GAT_name4 GATv2Conv --GAT_name5 GATv2Conv \
--activ_fun3 relu6 --activ_fun4 relu6 --activ_fun5 relu6 \
--squeeze_type conv \
--excite_type fc"

# 定义 top-4 组合（顺序无关）
declare -A TARGET_COMBINATIONS=(
    # ["GATConv_GATConv_rrelu_relu6"]="GATConv GATConv rrelu relu6"
    # ["GATConv_GATv2Conv_relu6_relu6"]="GATConv GATv2Conv relu6 relu6"
    # ["GATv2Conv_GATConv_relu6_rrelu"]="GATv2Conv GATConv relu6 rrelu"
    # ["GATv2Conv_GATv2Conv_leaky_relu_leaky_relu"]="GATv2Conv GATv2Conv leaky_relu leaky_relu"

    ["GATConv_GATConv_celu_leaky_relu"]="GATConv GATConv celu leaky_relu"
    ["GATConv_GATConv_elu_leaky_relu"]="GATConv GATConv elu leaky_relu"
    ["GATConv_GATv2Conv_leaky_relu_celu"]="GATConv GATv2Conv leaky_relu celu"
    ["GATConv_GATv2Conv_leaky_relu_selu"]="GATConv GATv2Conv leaky_relu selu"
    ["GATConv_GATv2Conv_leaky_relu_elu"]="GATConv GATv2Conv leaky_relu elu"
    ["GATv2Conv_GATConv_silu_selu"]="GATv2Conv GATConv silu selu"
    ["GATv2Conv_GATConv_silu_relu6"]="GATv2Conv GATConv silu relu6"
    ["GATv2Conv_GATConv_relu6_silu"]="GATv2Conv GATConv relu6 silu"
    ["GATv2Conv_GATv2Conv_leaky_relu_silu"]="GATv2Conv GATv2Conv leaky_relu silu"
    ["GATv2Conv_GATv2Conv_relu6_rrelu"]="GATv2Conv GATv2Conv relu6 rrelu"
)

echo "🔍 Checking and processing top-4 combinations with conv-fc SE..."

for ID in "${!TARGET_COMBINATIONS[@]}"; do
    echo "============================================"
    echo "Processing: $ID"
    echo "============================================"

    read gat1 gat2 act1 act2 <<< "${TARGET_COMBINATIONS[$ID]}"

    # 构建模型路径（与 tree 输出一致）
    MODEL_DIR="/root/shared-nvme/work/code/RCA/IncluRCA/model/a3/checkpoint/conv_fc_${gat1}_${gat2}_${act1}_${act2}"
    MODEL_PATH="$MODEL_DIR/main.pt"
    TRAIN_LOG="/root/shared-nvme/work/code/RCA/IncluRCA/code/logs/train_${ID}_conv_fc.log"
    LOCALIZER_LOG="/root/shared-nvme/work/code/RCA/IncluRCA/code/logs/localizer_${ID}_conv_fc.log"

    mkdir -p "$(dirname "$TRAIN_LOG")" "$(dirname "$LOCALIZER_LOG")"

    if [ -f "$MODEL_PATH" ]; then
        echo "✅ Model already exists at: $MODEL_PATH"
        echo "🔹 Skipping training, proceeding to localizer..."
    else
        echo "⚠️ Model NOT found. Training now..."
        mkdir -p "$MODEL_DIR"
        $BASE_TRAIN_CMD \
            --model_path "$MODEL_PATH" \
            --GAT_name1 "$gat1" \
            --GAT_name2 "$gat2" \
            --activ_fun1 "$act1" \
            --activ_fun2 "$act2" \
            | tee "$TRAIN_LOG"
        echo "✅ Training completed for $ID"
    fi

    # 总是运行 localizer（即使模型已存在）
    echo "🔹 Running a_localizer..."
    echo "a_localizer" | tee "$LOCALIZER_LOG"
    echo "feature_integration_MultiScaleConvSEAttention conv fc" | tee -a "$LOCALIZER_LOG"

    CMD_LOCALIZER="$BASE_LOCALIZER_CMD --model_path \"$MODEL_PATH\" --GAT_name1 \"$gat1\" --GAT_name2 \"$gat2\" --activ_fun1 \"$act1\" --activ_fun2 \"$act2\""
    eval $CMD_LOCALIZER | tee -a "$LOCALIZER_LOG"

    echo "✅ Localizer finished → $LOCALIZER_LOG"
    echo ""
done

echo "🎉 All top-4 combinations processed!"