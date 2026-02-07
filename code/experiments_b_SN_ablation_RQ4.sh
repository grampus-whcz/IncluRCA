#！/bin/bash
# nohup bash experiments_b_SN_ablation_RQ4.sh >> experiments_b_SN_ablation_RQ4.log 2>&1 &

# Main
echo "---Main Experiment---"
echo "---Main Experiment  b_SN_rca_trainer---"

echo "--window_size 8"

echo "# Training: new transformed data: trace, metric, log, api"
echo "reduction=4"
python ./IncluRCA/trainer/b_rca_trainer.py --window_size 8 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
--dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2023_Eadro_SN/dataset/merge/window_size_8.pkl \
--model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/b/checkpoint/main.pt \
--epochs 250 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 \
--GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu
