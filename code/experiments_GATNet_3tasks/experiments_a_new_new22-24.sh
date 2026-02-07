#！/bin/bash
# nohup bash experiments_a_new_new22-24.sh >> experiments_a_new_new22-24.log 2>&1 &

# Main
echo "---Main Experiment---"
echo "---Main Experiment  a_rca_trainer---"
echo "# new transformed data: trace, metric, log, api"
echo "# head 4, --orl_te_in_channels 512"
echo "# SEattention"
echo "# deterministic"

echo "22start############################"
# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a22/checkpoint/main.pt --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 celu --activ_fun2 mish
python ./IncluRCA/trainer/a_localizer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a22/checkpoint/main.pt --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 celu --activ_fun2 mish
echo "#############################end22"

echo "23start############################"
# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a23/checkpoint/main.pt --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 celu --activ_fun2 hardswish
python ./IncluRCA/trainer/a_localizer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a23/checkpoint/main.pt --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 celu --activ_fun2 hardswish
echo "#############################end23"

echo "24start############################"
# python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a24/checkpoint/main.pt --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 celu --activ_fun2 rrelu
python ./IncluRCA/trainer/a_localizer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a24/checkpoint/main.pt --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 celu --activ_fun2 rrelu
echo "#############################end24"


