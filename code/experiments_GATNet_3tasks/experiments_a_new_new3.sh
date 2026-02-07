#！/bin/bash
# nohup bash experiments_a_new_new3.sh >> experiments_a_new_new7-9.log 2>&1 &

# Main
echo "---Main Experiment---"
echo "---Main Experiment  a_rca_trainer---"
echo "# new transformed data: trace, metric, log, api"
echo "# head 4, --orl_te_in_channels 512"
echo "# SEattention"
echo "# deterministic"

echo "7start############################"
# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a7/checkpoint/main.pt --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 elu --activ_fun2 elu
python ./IncluRCA/trainer/a_localizer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a7/checkpoint/main.pt --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 elu --activ_fun2 elu
echo "#############################end7"

echo "8start############################"
# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a8/checkpoint/main.pt --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 silu
python ./IncluRCA/trainer/a_localizer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a8/checkpoint/main.pt --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 silu
echo "#############################end8"

echo "9start############################"
# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a9/checkpoint/main.pt --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 tanhshrink --activ_fun2 leaky_relu
python ./IncluRCA/trainer/a_localizer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a9/checkpoint/main.pt --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 tanhshrink --activ_fun2 leaky_relu
echo "#############################end9"

