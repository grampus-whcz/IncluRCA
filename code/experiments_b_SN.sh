#！/bin/bash
# nohup bash experiments_b_SN.sh >> experiments_b_SN.log 2>&1 &

# Main
echo "---Main Experiment---"
echo "---Main Experiment  b_SN_rca_trainer---"

echo "--window_size 8"

echo "# Training: new transformed data: trace, metric, log, api"
python ./IncluRCA/trainer/b_rca_trainer.py --window_size 8 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
--dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2023_Eadro_SN/dataset/merge/window_size_8.pkl \
--model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/b/checkpoint/main.pt \
--epochs 250 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 \
--GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu


echo "# b_Localizer: "
python ./IncluRCA/trainer/b_localizer.py --window_size 8 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
--dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2023_Eadro_SN/dataset/merge/window_size_8.pkl \
--model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/b/checkpoint/main.pt \
--epochs 250 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 \
--GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu



# original
echo "# transformer IncluRCA original data"
python ./IncluRCA/trainer/b_rca_trainer.py --window_size 8 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
--dataset_path /root/shared-nvme/work/code/RCA/data_set/IncluRCATempData-main/temp_data_split/temp_data/2023_Eadro_SN/dataset/merge/window_size_8.pkl \
--model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/b/checkpoint/main.pt \
--epochs 250 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 128 --efi_in_dim 128 --efi_te_heads 2 --efi_te_layers 1 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.1 \
--GAT_name1 GATv2Conv --GAT_name2 GATv2Conv --activ_fun1 elu --activ_fun2 elu --explainer_mask_acti_func relu

echo "# IncluRCA  b_Localizer: "
python ./IncluRCA/trainer/b_localizer.py --window_size 8 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
--dataset_path /root/shared-nvme/work/code/RCA/data_set/IncluRCATempData-main/temp_data_split/temp_data/2023_Eadro_SN/dataset/merge/window_size_8.pkl \
--model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/b/checkpoint/main.pt \
--epochs 250 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 128 --efi_in_dim 128 --efi_te_heads 2 --efi_te_layers 1 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.1 \
--GAT_name1 GATv2Conv --GAT_name2 GATv2Conv --activ_fun1 elu --activ_fun2 elu --explainer_mask_acti_func relu

