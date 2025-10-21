#！/bin/bash
# nohup bash experiments_d.sh >> experiments_d.log 2>&1 &

echo "--window_size 17"
echo "# IncluRCA Training: trace, metric, log, api"
python ./IncluRCA/trainer/a_rca_trainer.py --window_size 17 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
--dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2025_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_17.pkl \
--model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/d/checkpoint/main.pt \
--epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 \
--ec_fault_types 14 \
--GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu

echo "--window_size 17"
echo "# IncluRCA Training: trace, metric, log, api"
echo "d_localizer"
python ./IncluRCA/trainer/a_localizer.py --window_size 17 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
--dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2025_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_17.pkl \
--model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/d/checkpoint/main.pt \
--epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 \
--GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu

echo "--window_size 17"
echo "# IncluRCA Training: trace, metric, log"
python ./IncluRCA/trainer/a_rca_trainer.py --window_size 17 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
--dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2025_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_17.pkl \
--model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/d/checkpoint/main.pt \
--epochs 300 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 256 --efi_in_dim 256 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --batch_size 64  --eff_GAT_dropout 0.1 \
--ec_fault_types 14 \
--GAT_name1 GATv2Conv --GAT_name2 GATv2Conv --activ_fun1 elu --activ_fun2 elu --explainer_mask_acti_func relu

echo "--window_size 17"
echo "# IncluRCA Training: trace, metric, log"
echo "d_localizer"
python ./IncluRCA/trainer/a_localizer.py --window_size 17 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
--dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2025_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_17.pkl \
--model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/d/checkpoint/main.pt \
--epochs 300 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 256 --efi_in_dim 256 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.1 \
--GAT_name1 GATv2Conv --GAT_name2 GATv2Conv --activ_fun1 elu --activ_fun2 elu --explainer_mask_acti_func relu