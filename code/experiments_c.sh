#！/bin/bash
# nohup bash experiments_c.sh >> experiments_c.log 2>&1 &

# original
echo "# transformer IncluRCA original data epochs 100"
python ./IncluRCA/trainer/c_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/c/checkpoint/main.pt \
--epochs 100 --orl_te_heads 2 --orl_te_layers 2 --orl_te_in_channels 128 --efi_in_dim 128 --efi_te_heads 4 --efi_te_layers 2 --efi_out_dim 256 --eff_in_dim 256 --eff_GAT_out_channels 128 --eff_GAT_heads 2 --eff_GAT_dropout 0.4 \
--GAT_name1 GATv2Conv --GAT_name2 GATv2Conv --activ_fun1 elu --activ_fun2 elu --explainer_mask_acti_func relu


echo "# SEattention IncluRCA original data epochs 100"
python ./IncluRCA/trainer/c_rca_trainer.py --model_path /root/shared-nvme/work/code/RCA/IncluRCA/model/c/checkpoint/main.pt \
--epochs 100 --orl_te_heads 4 --orl_te_layers 2 --orl_te_in_channels 512 --efi_in_dim 512 --efi_te_heads 8 --efi_te_layers 2 --efi_out_dim 512 --eff_in_dim 512 --eff_GAT_out_channels 128 --eff_GAT_heads 4 --eff_GAT_dropout 0.4 \
--GAT_name1 GATv2Conv --GAT_name2 GATConv --activ_fun1 relu6 --activ_fun2 relu6 --explainer_mask_acti_func relu