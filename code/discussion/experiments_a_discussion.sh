#！/bin/bash
# nohup bash experiments_a_discussion.sh >> experiments_a_discussion_parameters.log 2>&1 &

echo "IncluRCA a_rca_trainer old fusion"
echo "feature_integration_MultiScaleConvSEAttention reduction=32"
echo "--squeeze_type max --excite_type conv"
python ./IncluRCA/trainer/a_rca_trainer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
--dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
--model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt \
--epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 \
--GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 celu --explainer_mask_acti_func relu \
--GAT_name3 GATv2Conv --GAT_name4 GATConv --GAT_name5 GATConv --activ_fun3 relu6 --activ_fun4 relu6 --activ_fun5 relu6 \
--squeeze_type max --excite_type conv

echo "a_localizer"
echo "feature_integration_MultiScaleConvSEAttention max conv reduction=32"
python ./IncluRCA/trainer/a_localizer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
--dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
--model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt \
--epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 \
--GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 celu --explainer_mask_acti_func relu \
--GAT_name3 GATv2Conv --GAT_name4 GATConv --GAT_name5 GATv2Conv --activ_fun3 elu --activ_fun4 elu --activ_fun5 elu \
--squeeze_type max --excite_type conv

# echo "IncluRCA a_rca_trainer old fusion"
# echo "feature_integration_MultiScaleConvSEAttention"
# echo "--squeeze_type conv --excite_type fc"
# python ./IncluRCA/trainer/a_rca_trainer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
# --dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
# --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt \
# --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 \
# --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu \
# --GAT_name3 GATv2Conv --GAT_name4 GATConv --GAT_name5 GATConv --activ_fun3 relu6 --activ_fun4 relu6 --activ_fun5 relu6 \
# --squeeze_type conv --excite_type fc

# echo "a_localizer"
# echo "feature_integration_MultiScaleConvSEAttention conv fc reduction=32"
# python ./IncluRCA/trainer/a_localizer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
# --dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
# --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt \
# --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 \
# --GAT_name1 GATConv --GAT_name2 GATv2Conv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu \
# --GAT_name3 GATv2Conv --GAT_name4 GATv2Conv --GAT_name5 GATv2Conv --activ_fun3 elu --activ_fun4 elu --activ_fun5 elu \
# --squeeze_type conv --excite_type fc

## SEAttention
# echo "IncluRCA a_rca_trainer old fusion"
# echo "feature_integration_MultiScaleConvSEAttention"
# echo "--squeeze_type avg --excite_type fc"
# python ./IncluRCA/trainer/a_rca_trainer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
# --dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
# --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt \
# --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 \
# --GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu \
# --GAT_name3 GATv2Conv --GAT_name4 GATConv --GAT_name5 GATConv --activ_fun3 relu6 --activ_fun4 relu6 --activ_fun5 relu6 \
# --squeeze_type avg --excite_type fc

# echo "a_localizer"
# echo "feature_integration_MultiScaleConvSEAttention avg fc reduction=16"
# python ./IncluRCA/trainer/a_localizer.py --window_size 11 --data_base_path /root/shared-nvme/work/code/RCA/IncluRCA \
# --dataset_path /root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_11.pkl \
# --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/a/checkpoint/main.pt \
# --epochs 300 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.1 \
# --GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu \
# --GAT_name3 GATv2Conv --GAT_name4 GATConv --GAT_name5 GATConv --activ_fun3 elu --activ_fun4 elu --activ_fun5 elu \
# --squeeze_type avg --excite_type fc



