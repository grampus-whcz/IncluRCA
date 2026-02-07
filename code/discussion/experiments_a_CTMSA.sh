#！/bin/bash
# nohup bash experiments_a_CTMSA.sh >> experiments_a_parameters.log 2>&1 &

echo "IncluRCA feature_integration_CTMSA --window_size 11"
echo "--GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 relu6 --activ_fun2 selu"
python ./IncluRCA/trainer/a_rca_trainer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
--dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
--model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a2/checkpoint/main.pt \
--epochs 300 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 256 --efi_in_dim 256 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --batch_size 64 --eff_GAT_dropout 0.1 \
--GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 relu6 --activ_fun2 selu --explainer_mask_acti_func relu \
--GAT_name3 GATv2Conv --GAT_name4 GATConv --GAT_name5 GATConv --activ_fun3 relu6 --activ_fun4 relu6 --activ_fun5 relu6 \
--squeeze_type avg --excite_type fc

echo "IncluRCA a_localizer  feature_integration_CTMSA --window_size 11"
python ./IncluRCA/trainer/a_localizer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
--dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
--model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a2/checkpoint/main.pt \
--epochs 300 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 256 --efi_in_dim 256 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.1 \
--GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 relu6 --activ_fun2 selu --explainer_mask_acti_func relu \
--GAT_name3 GATv2Conv --GAT_name4 GATv2Conv --GAT_name5 GATv2Conv --activ_fun3 relu6 --activ_fun4 relu6 --activ_fun5 relu6 \
--squeeze_type avg --excite_type fc

# echo "IncluRCA feature_integration_CTMSA --window_size 11"
# echo "--GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 selu --activ_fun2 celu"
# python ./IncluRCA/trainer/a_rca_trainer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
# --dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
# --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a2/checkpoint/main.pt \
# --epochs 300 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 256 --efi_in_dim 256 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --batch_size 64 --eff_GAT_dropout 0.1 \
# --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 selu --activ_fun2 celu --explainer_mask_acti_func relu \
# --GAT_name3 GATv2Conv --GAT_name4 GATConv --GAT_name5 GATConv --activ_fun3 relu6 --activ_fun4 relu6 --activ_fun5 relu6 \
# --squeeze_type avg --excite_type fc


# echo "IncluRCA a_localizer  feature_integration_CTMSA --window_size 11"
# python ./IncluRCA/trainer/a_localizer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
# --dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
# --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a2/checkpoint/main.pt \
# --epochs 300 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 256 --efi_in_dim 256 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.1 \
# --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 selu --activ_fun2 celu --explainer_mask_acti_func relu \
# --GAT_name3 GATv2Conv --GAT_name4 GATv2Conv --GAT_name5 GATv2Conv --activ_fun3 relu6 --activ_fun4 relu6 --activ_fun5 relu6 \
# --squeeze_type avg --excite_type fc


# echo "IncluRCA feature_integration_CTMSA --window_size 11"
# echo "--GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 selu --activ_fun2 elu"
# python ./IncluRCA/trainer/a_rca_trainer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
# --dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
# --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a2/checkpoint/main.pt \
# --epochs 300 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 256 --efi_in_dim 256 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --batch_size 64 --eff_GAT_dropout 0.1 \
# --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 selu --activ_fun2 elu --explainer_mask_acti_func relu \
# --GAT_name3 GATv2Conv --GAT_name4 GATConv --GAT_name5 GATConv --activ_fun3 relu6 --activ_fun4 relu6 --activ_fun5 relu6 \
# --squeeze_type avg --excite_type fc


# echo "IncluRCA a_localizer  feature_integration_CTMSA --window_size 11"
# python ./IncluRCA/trainer/a_localizer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
# --dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
# --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a2/checkpoint/main.pt \
# --epochs 300 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 256 --efi_in_dim 256 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.1 \
# --GAT_name1 GATConv --GAT_name2 GATConv --activ_fun1 selu --activ_fun2 elu --explainer_mask_acti_func relu \
# --GAT_name3 GATv2Conv --GAT_name4 GATv2Conv --GAT_name5 GATv2Conv --activ_fun3 relu6 --activ_fun4 relu6 --activ_fun5 relu6 \
# --squeeze_type avg --excite_type fc


# echo "IncluRCA feature_integration_CTMSA --window_size 11"
# echo "--GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6"
# python ./IncluRCA/trainer/a_rca_trainer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
# --dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
# --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a2/checkpoint/main.pt \
# --epochs 300 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 256 --efi_in_dim 256 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --batch_size 64 --eff_GAT_dropout 0.1 \
# --GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu \
# --GAT_name3 GATv2Conv --GAT_name4 GATConv --GAT_name5 GATConv --activ_fun3 relu6 --activ_fun4 relu6 --activ_fun5 relu6 \
# --squeeze_type avg --excite_type fc


# echo "IncluRCA a_localizer  feature_integration_CTMSA --window_size 11"
# python ./IncluRCA/trainer/a_localizer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
# --dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
# --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a2/checkpoint/main.pt \
# --epochs 300 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 256 --efi_in_dim 256 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.1 \
# --GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu \
# --GAT_name3 GATv2Conv --GAT_name4 GATv2Conv --GAT_name5 GATv2Conv --activ_fun3 relu6 --activ_fun4 relu6 --activ_fun5 relu6 \
# --squeeze_type avg --excite_type fc

