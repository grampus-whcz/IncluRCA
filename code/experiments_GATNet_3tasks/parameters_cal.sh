#！/bin/bash
# nohup bash experiments_a_new1.sh >> experiments_a_new1.log 2>&1 &

# Main
echo "---Main Experiment---"
echo "---Main Experiment  a_rca_trainer---"
echo "# new transformed data: trace, metric, log, api"
echo "# head 4, --orl_te_in_channels 512"
echo "# SEattention"
echo "# deterministic"


# incluRCA parameters volume study
# echo "start############################"
python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt --epochs 20 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu

# IncluRCA parameters volume study
echo "# original: trace, metric, log"
echo "# head 2, --orl_te_in_channels 256"
echo "# transformer"
echo "# deterministic"

python ./IncluRCA/trainer/a_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt --epochs 10 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 256 --efi_in_dim 256 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --batch_size 64 --eff_GAT_dropout 0.1 --GAT_name1 GATv2Conv --GAT_name2 GATv2Conv --activ_fun1 elu --activ_fun2 elu --explainer_mask_acti_func relu
