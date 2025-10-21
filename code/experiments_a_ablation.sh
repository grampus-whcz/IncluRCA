#！/bin/bash
# nohup bash experiments_a_ablation.sh >> experiments_a_ablation.log 2>&1 &

# Main
echo "---Main Experiment ablation---"
echo "---Main Experiment  a_rca_trainer---"
echo "# head 4, --orl_te_in_channels 512"
echo "# SEattention"
echo "# deterministic"

echo "# new transformed data: trace, metric, log, api"
python ./IncluRCA/trainer/a_rca_trainer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/Repdf \
--dataset_path /root/shared-nvme/work/code/Repdf/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
--model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt \
--epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 \
--GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu

echo "# new transformed data: metric, log, api"
python ./IncluRCA/trainer/a_rca_trainer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/Repdf \
--dataset_path /root/shared-nvme/work/code/Repdf/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11_no_trace.pkl \
--model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt \
--epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 \
--GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu

echo "# new transformed data: metric, api"
python ./IncluRCA/trainer/a_rca_trainer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/Repdf \
--dataset_path /root/shared-nvme/work/code/Repdf/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11_no_trace_no_log.pkl \
--model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt \
--epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 \
--GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu

echo "# new transformed data: trace, metric, api"
python ./IncluRCA/trainer/a_rca_trainer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/Repdf \
--dataset_path /root/shared-nvme/work/code/Repdf/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11_no_log.pkl \
--model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt \
--epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 \
--GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu

echo "# new transformed data: trace, metric, log"
python ./IncluRCA/trainer/a_rca_trainer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/Repdf \
--dataset_path /root/shared-nvme/work/code/Repdf/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11_no_api.pkl \
--model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt \
--epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 \
--GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu

echo "# Holistic data: trace, metric, log"
python ./IncluRCA/trainer/a_rca_trainer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
--dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
--model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt \
--epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 \
--GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu


echo "# new transformed data: trace, metric, log"
python ./IncluRCA/trainer/a_rca_trainer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/Repdf \
--dataset_path /root/shared-nvme/work/code/Repdf/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11_no_api.pkl \
--model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt \
--epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --batch_size 64 --eff_GAT_dropout 0.1 \
--GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu

