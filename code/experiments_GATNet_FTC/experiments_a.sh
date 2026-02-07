#！/bin/bash
# nohup bash experiments_a.sh >> experiments_a.log 2>&1 &

# Main
echo "---Main Experiment---"

echo "---Main Experiment  a_rca_trainer---"
# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 256 --efi_in_dim 256 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.1

echo "---Main Experiment  a_rca_trainer---"

echo "# new transformed data: trace, metric, log, api"
echo "# head 4, --orl_te_in_channels 512"
# echo "# old data"
echo "# SEattention"
echo "# deterministic"

# GATConv, GATv2Conv
# hardtanh, mish, hardswish, silu, tanh, hardsigmoid, rrelu, leaky_relu, celu, selu, elu, relu6, tanhshrink

# 定义参数列表
GAT_NAMES=("GATConv" "GATv2Conv")
ACTIV_FUNS=("hardtanh" "mish" "hardswish" "silu" "tanh" "hardsigmoid" "rrelu" "leaky_relu" "celu" "selu" "elu" "relu6" "tanhshrink")


# super parameters in IncluRCA
# --epochs 300 \
# --orl_te_heads 2 \
# --orl_te_layers 2 \
# --orl_te_in_channels 256 \
# --efi_in_dim 256 \
# --efi_te_heads 4 \
# --efi_te_layers 2 \
# --efi_out_dim 256 \
# --eff_in_dim 256 \
# --eff_GAT_out_channels 128 \
# --eff_GAT_heads 2 \
# --batch_size 64 \
# --eff_GAT_dropout 0.1"

# 基础命令（不包含要遍历的参数）
# IncluRCA super parameters
BASE_CMD="python ../IncluRCA/trainer/a_rca_trainer.py \
  --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt \
  --epochs 300 \
  --orl_te_heads 4 \
  --orl_te_layers 2 \
  --orl_te_in_channels 512 \
  --efi_in_dim 512 \
  --efi_te_heads 8 \
  --efi_te_layers 2 \
  --efi_out_dim 512 \
  --eff_in_dim 512 \
  --eff_GAT_out_channels 128 \
  --eff_GAT_heads 4 \
  --batch_size 64 \
  --eff_GAT_dropout 0.1"


  

# 遍历所有组合
for gat1 in "${GAT_NAMES[@]}"; do
  for gat2 in "${GAT_NAMES[@]}"; do
    for act1 in "${ACTIV_FUNS[@]}"; do
      for act2 in "${ACTIV_FUNS[@]}"; do
        # 构造完整命令
        CMD="$BASE_CMD \
          --GAT_name1 $gat1 \
          --GAT_name2 $gat2 \
          --activ_fun1 $act1 \
          --activ_fun2 $act2"

        echo "begin----------------------------------------"
        # 打印当前命令
        echo "Running: $CMD"

        # 执行命令（取消注释下面这行以实际运行）
        eval $CMD

        # 如果你想后台运行，可以使用：
        # eval $CMD &
      done
    done
  done
done

# 等待所有后台任务完成（如果使用了 &）
# wait














# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 hardtanh --activ_fun2 hardtanh

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 mish --activ_fun2 mish

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 hardswish --activ_fun2 hardswish

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 silu --activ_fun2 silu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 tanh --activ_fun2 tanh

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 hardsigmoid --activ_fun2 hardsigmoid

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 rrelu --activ_fun2 rrelu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 leaky_relu --activ_fun2 leaky_relu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 celu --activ_fun2 celu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 selu --activ_fun2 selu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 elu --activ_fun2 elu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 tanhshrink --activ_fun2 tanhshrink


# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATv2Conv --activ_fun1 hardtanh --activ_fun2 hardtanh

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATv2Conv --activ_fun1 mish --activ_fun2 mish

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATv2Conv --activ_fun1 hardswish --activ_fun2 hardswish

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATv2Conv --activ_fun1 silu --activ_fun2 silu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATv2Conv --activ_fun1 tanh --activ_fun2 tanh

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATv2Conv --activ_fun1 hardsigmoid --activ_fun2 hardsigmoid

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATv2Conv --activ_fun1 rrelu --activ_fun2 rrelu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATv2Conv --activ_fun1 leaky_relu --activ_fun2 leaky_relu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATv2Conv --activ_fun1 celu --activ_fun2 celu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATv2Conv --activ_fun1 selu --activ_fun2 selu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATv2Conv --activ_fun1 elu --activ_fun2 elu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATv2Conv --activ_fun1 relu6 --activ_fun2 relu6

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATv2Conv --activ_fun1 tanhshrink --activ_fun2 tanhshrink


# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 hardtanh --activ_fun2 hardtanh

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 mish --activ_fun2 mish

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 hardswish --activ_fun2 hardswish

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 silu --activ_fun2 silu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 tanh --activ_fun2 tanh

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 hardsigmoid --activ_fun2 hardsigmoid

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 rrelu --activ_fun2 rrelu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 leaky_relu --activ_fun2 leaky_relu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 celu --activ_fun2 celu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 selu --activ_fun2 selu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 elu --activ_fun2 elu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 relu6 --activ_fun2 relu6

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 tanhshrink --activ_fun2 tanhshrink


# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 hardtanh --activ_fun2 hardtanh

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 mish --activ_fun2 mish

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 hardswish --activ_fun2 hardswish

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 silu --activ_fun2 silu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 tanh --activ_fun2 tanh

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 hardsigmoid --activ_fun2 hardsigmoid

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 rrelu --activ_fun2 rrelu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 leaky_relu --activ_fun2 leaky_relu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 celu --activ_fun2 celu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 selu --activ_fun2 selu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 elu --activ_fun2 elu

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6

# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt  --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 tanhshrink --activ_fun2 tanhshrink



# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt --epochs 300 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 256 --efi_in_dim 256 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.1

# python ./IncluRCA/trainer/a_localizer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt --epochs 300 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 256 --efi_in_dim 256 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.1

# # Ablation Study
# echo "--Ablation Study 1---"
# python ./IncluRCA/ablation/a_ablation1_trainer.py
# echo "--Ablation Study 2---"
# python ./IncluRCA/ablation/a_ablation2_trainer.py
# echo "--Ablation Study 3---"
# python ./IncluRCA/ablation/a_ablation3_trainer.py

# # Sensitivity Analysis
# echo "---------------------"
# echo "orl_te_in_channels: 32"
# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/orl_te_in_channels_32.pt --epochs 300 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 32 --efi_in_dim 32 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.1
# echo "---------------------"
# echo "orl_te_in_channels: 64"
# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/orl_te_in_channels_64.pt --epochs 300 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 64 --efi_in_dim 64 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.1
# echo "---------------------"
# echo "orl_te_in_channels: 128"
# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/orl_te_in_channels_128.pt --epochs 300 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 128 --efi_in_dim 128 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.1
# echo "---------------------"

# echo "---------------------"
# echo "eff_in_dim: 32"
# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/eff_in_dim_32.pt --epochs 300 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 256 --efi_in_dim 256 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 32 --eff_in_dim 32 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.1
# echo "---------------------"e
# cho "eff_in_dim: 64"
# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/eff_in_dim_64.pt --epochs 300 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 256 --efi_in_dim 256 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 64 --eff_in_dim 64 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.1
# echo "---------------------"
# echo "eff_in_dim: 128"
# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/eff_in_dim_128.pt --epochs 300 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 256 --efi_in_dim 256 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 128 --eff_in_dim 128 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.1
# echo "---------------------"
