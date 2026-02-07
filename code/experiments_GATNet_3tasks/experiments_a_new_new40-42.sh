#！/bin/bash
# nohup bash experiments_a_new_new40-42.sh >> experiments_a_new_new40-42.log 2>&1 &

# Main
echo "---Main Experiment---"
echo "---Main Experiment  a_rca_trainer---"
echo "# new transformed data: trace, metric, log, api"
echo "# head 4, --orl_te_in_channels 512"
echo "# SEattention"
echo "# deterministic"

echo "40start############################"
# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a40/checkpoint/main.pt --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 silu --activ_fun2 selu
python ./IncluRCA/trainer/a_localizer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a40/checkpoint/main.pt --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 silu --activ_fun2 selu
echo "#############################end40"

echo "41start############################"
# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a41/checkpoint/main.pt --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 rrelu --activ_fun2 leaky_relu
python ./IncluRCA/trainer/a_localizer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a41/checkpoint/main.pt --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 rrelu --activ_fun2 leaky_relu
echo "#############################end41"

echo "42start############################"
# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a42/checkpoint/main.pt --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 leaky_relu --activ_fun2 leaky_relu
python ./IncluRCA/trainer/a_localizer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a42/checkpoint/main.pt --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 leaky_relu --activ_fun2 leaky_relu
echo "#############################end42"


