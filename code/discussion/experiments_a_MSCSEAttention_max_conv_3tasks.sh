#!/bin/bash
# nohup bash experiments_a_MSCSEAttention_max_conv_3tasks.sh >> experiments_a_MSCSEAttention_max_conv_3tasks.log 2>&1 &


# 重跑已筛选出的3个最佳组合，并在每次训练后运行 a_localizer.py
set -e  # 遇到错误立即退出

# 基础训练命令（与原脚本一致）
BASE_TRAIN_CMD="python ./IncluRCA/trainer/a_rca_trainer.py \
--window_size 11 \
--data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
--dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
--epochs 300 \
--orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 \
--efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 \
--eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 \
--GAT_name3 GATv2Conv --GAT_name4 GATConv --GAT_name5 GATConv \
--activ_fun3 relu6 --activ_fun4 relu6 --activ_fun5 relu6"

# Localizer 命令模板（注意：model_path 会动态替换）
BASE_LOCALIZER_CMD="python ./IncluRCA/trainer/a_localizer.py \
--window_size 11 \
--data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
--dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
--epochs 300 \
--orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 \
--efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 \
--eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 \
--explainer_mask_acti_func relu \
--GAT_name3 GATv2Conv --GAT_name4 GATv2Conv --GAT_name5 GATv2Conv --activ_fun3 relu6 --activ_fun4 relu6 --activ_fun5 relu6 \
--squeeze_type max --excite_type conv"

# 定义最佳组合（来自你的筛选结果）
# ["max_conv_GATv2Conv_GATConv_silu_rrelu"]="GATv2Conv GATConv silu rrelu"
declare -A BEST_COMBINATIONS=(    
    ["max_conv_GATv2Conv_GATv2Conv_relu6_selu"]="GATv2Conv GATv2Conv relu6 selu"
    ["max_conv_GATConv_GATv2Conv_silu_silu"]="GATConv GATv2Conv silu silu"
    ["max_conv_GATConv_GATv2Conv_leaky_relu_silu"]="GATConv GATv2Conv leaky_relu silu"
    ["max_conv_GATConv_GATv2Conv_leaky_relu_rrelu"]="GATConv GATv2Conv leaky_relu rrelu"
    ["max_conv_GATConv_GATv2Conv_leaky_relu_celu"]="GATConv GATv2Conv leaky_relu celu"
    ["max_conv_GATConv_GATv2Conv_celu_celu"]="GATConv GATv2Conv celu celu"
    ["max_conv_GATConv_GATv2Conv_celu_selu"]="GATConv GATv2Conv celu selu"
    ["max_conv_GATConv_GATv2Conv_celu_elu"]="GATConv GATv2Conv celu elu"
    ["max_conv_GATConv_GATv2Conv_selu_leaky_relu"]="GATConv GATv2Conv selu leaky_relu"
    ["max_conv_GATConv_GATv2Conv_elu_celu"]="GATConv GATv2Conv elu celu"
    ["max_conv_GATConv_GATv2Conv_elu_selu"]="GATConv GATv2Conv elu selu"
    ["max_conv_GATConv_GATv2Conv_elu_elu"]="GATConv GATv2Conv elu elu"
    ["max_conv_GATConv_GATv2Conv_relu6_silu"]="GATConv GATv2Conv relu6 silu"
    ["max_conv_GATConv_GATv2Conv_relu6_celu"]="GATConv GATv2Conv relu6 celu"    
    ["max_conv_GATv2Conv_GATConv_silu_leaky_relu"]="GATv2Conv GATConv silu leaky_relu"
    ["max_conv_GATv2Conv_GATConv_rrelu_silu"]="GATv2Conv GATConv rrelu silu"
    ["max_conv_GATv2Conv_GATConv_rrelu_leaky_relu"]="GATv2Conv GATConv rrelu leaky_relu"
    ["max_conv_GATv2Conv_GATConv_rrelu_celu"]="GATv2Conv GATConv rrelu celu"
    ["max_conv_GATv2Conv_GATConv_rrelu_elu"]="GATv2Conv GATConv rrelu elu"
    ["max_conv_GATv2Conv_GATConv_rrelu_relu6"]="GATv2Conv GATConv rrelu relu6"
    ["max_conv_GATv2Conv_GATConv_leaky_relu_relu6"]="GATv2Conv GATConv leaky_relu relu6"
    ["max_conv_GATv2Conv_GATConv_celu_celu"]="GATv2Conv GATConv celu celu"
    ["max_conv_GATv2Conv_GATConv_celu_elu"]="GATv2Conv GATConv celu elu"
    ["max_conv_GATv2Conv_GATConv_selu_silu"]="GATv2Conv GATConv selu silu"
    ["max_conv_GATv2Conv_GATConv_selu_rrelu"]="GATv2Conv GATConv selu rrelu"
    ["max_conv_GATv2Conv_GATConv_selu_leaky_relu"]="GATv2Conv GATConv selu leaky_relu"
    ["max_conv_GATv2Conv_GATConv_elu_celu"]="GATv2Conv GATConv elu celu"
    ["max_conv_GATv2Conv_GATConv_elu_elu"]="GATv2Conv GATConv elu elu"
    ["max_conv_GATv2Conv_GATConv_relu6_silu"]="GATv2Conv GATConv relu6 silu"
    ["max_conv_GATv2Conv_GATConv_relu6_celu"]="GATv2Conv GATConv relu6 celu"
    ["max_conv_GATv2Conv_GATConv_relu6_elu"]="GATv2Conv GATConv relu6 elu"
    ["max_conv_GATv2Conv_GATConv_relu6_relu6"]="GATv2Conv GATConv relu6 relu6"
    ["max_conv_GATv2Conv_GATv2Conv_silu_rrelu"]="GATv2Conv GATv2Conv silu rrelu"
    ["max_conv_GATv2Conv_GATv2Conv_silu_leaky_relu"]="GATv2Conv GATv2Conv silu leaky_relu"
    ["max_conv_GATv2Conv_GATv2Conv_silu_selu"]="GATv2Conv GATv2Conv silu selu"
    ["max_conv_GATv2Conv_GATv2Conv_rrelu_silu"]="GATv2Conv GATv2Conv rrelu silu"
    ["max_conv_GATv2Conv_GATv2Conv_rrelu_rrelu"]="GATv2Conv GATv2Conv rrelu rrelu"
    ["max_conv_GATv2Conv_GATv2Conv_rrelu_relu6"]="GATv2Conv GATv2Conv rrelu relu6"
    ["max_conv_GATv2Conv_GATv2Conv_leaky_relu_leaky_relu"]="GATv2Conv GATv2Conv leaky_relu leaky_relu"
    ["max_conv_GATv2Conv_GATv2Conv_leaky_relu_selu"]="GATv2Conv GATv2Conv leaky_relu selu"
    ["max_conv_GATv2Conv_GATv2Conv_leaky_relu_relu6"]="GATv2Conv GATv2Conv leaky_relu relu6"
    ["max_conv_GATv2Conv_GATv2Conv_celu_elu"]="GATv2Conv GATv2Conv celu elu"
    ["max_conv_GATv2Conv_GATv2Conv_relu6_silu"]="GATv2Conv GATv2Conv relu6 silu"    
    ["max_conv_GATv2Conv_GATv2Conv_relu6_relu6"]="GATv2Conv GATv2Conv relu6 relu6"
)

echo "🚀 Starting re-run of top 3 best combinations..."

for ID in "${!BEST_COMBINATIONS[@]}"; do
    echo "============================================"
    echo "Running best combination: $ID"
    echo "============================================"

    # 解析参数
    read gat1 gat2 act1 act2 <<< "${BEST_COMBINATIONS[$ID]}"

    MODEL_DIR="/root/shared-nvme/work/code/RCA/IncluRCA/model/a3/checkpoint/${ID}"
    MODEL_PATH="$MODEL_DIR/main.pt"
    TRAIN_LOG="/root/shared-nvme/work/code/RCA/IncluRCA/code/logs/train_${ID}.log"
    LOCALIZER_LOG="/root/shared-nvme/work/code/RCA/IncluRCA/code/logs/localizer_${ID}.log"

    mkdir -p "$MODEL_DIR"

    # Step 1: 训练主模型
    echo "🔹 Training model..."
    $BASE_TRAIN_CMD \
        --model_path "$MODEL_PATH" \
        --squeeze_type "max" \
        --excite_type "conv" \
        --GAT_name1 "$gat1" \
        --GAT_name2 "$gat2" \
        --activ_fun1 "$act1" \
        --activ_fun2 "$act2" \
        | tee "$TRAIN_LOG"

    # echo "✅ Training finished for $ID"

    # Step 2: 运行 localizer（使用刚训练好的模型）
    echo "🔹 Running a_localizer with trained model..."
    echo "a_localizer" | tee -a "$LOCALIZER_LOG"
    echo "feature_integration_MultiScaleConvSEAttention avg conv reduction=32" | tee -a "$LOCALIZER_LOG"

    # 动态插入 model_path
    CMD_LOCALIZER="$BASE_LOCALIZER_CMD --model_path \"$MODEL_PATH\" --GAT_name1 \"$gat1\" --GAT_name2 \"$gat2\" --activ_fun1 \"$act1\" --activ_fun2 \"$act2\""
    eval $CMD_LOCALIZER | tee -a "$LOCALIZER_LOG"

    echo "✅ Localizer finished for $ID → logs saved to $LOCALIZER_LOG"
    echo ""
done

echo "🎉 All best combinations re-trained and localized successfully!"