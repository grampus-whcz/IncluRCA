import sys
sys.path.append('/root/shared-nvme/work/code/RCA/IncluRCA/code')
import torch

from IncluRCA.base.base_rca_trainer import BaseRCATrainer
from IncluRCA.explain.explainer import Explainer
from IncluRCA.util.data_handler import copy_batch_data, rearrange_y
from shared_util.evaluation_metrics import *
import argparse
from shared_util.file_handler import FileHandler
from shared_util.seed import *
import copy


class CRCATrainer(BaseRCATrainer):
    def __init__(self, param_dict):
        super().__init__(param_dict)


if __name__ == '__main__':
    seed_everything()
    torch.use_deterministic_algorithms(True)

    parser = argparse.ArgumentParser()

    # window_size = 6
    # data_base_path = '/root/shared-nvme/work/code/RCA/IncluRCA'
    # parser.add_argument("--dataset_path", default=f'{data_base_path}/temp_data/2023_Eadro_SN/dataset/merge/window_size_{window_size}.pkl', type=str)
    # model_base_path = FileHandler.set_folder(f'{data_base_path}/model/c')
    # parser.add_argument("--model_path", default=f'{FileHandler.set_folder(model_base_path + "/checkpoint")}/main.pt', type=str)
    # parser.add_argument("--window_size", default=window_size, type=int)

    parser.add_argument("--gpu", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--service_accuracy_th", default=0.2, type=float)

    parser.add_argument("--orl_te_heads", default=4, type=int)
    parser.add_argument("--orl_te_layers", default=2, type=int)
    parser.add_argument("--orl_te_in_channels", default=128, type=int)

    parser.add_argument("--efi_in_dim", default=128, type=int)
    parser.add_argument("--efi_te_heads", default=2, type=int)
    parser.add_argument("--efi_te_layers", default=1, type=int)
    parser.add_argument("--efi_out_dim", default=64 * 4, type=int)

    parser.add_argument("--eff_in_dim", default=64 * 4, type=int)
    parser.add_argument("--eff_GAT_out_channels", default=128, type=int)
    parser.add_argument("--eff_GAT_heads", default=2, type=int)
    parser.add_argument("--eff_GAT_dropout", default=0.1, type=float)

    parser.add_argument("--ec_fault_types", default=3, type=int)
    
    parser.add_argument("--GAT_name1", default="GATv2Conv", type=str)
    parser.add_argument("--GAT_name2", default="GATv2Conv", type=str)
    parser.add_argument("--activ_fun1", default="elu", type=str)
    parser.add_argument("--activ_fun2", default="elu", type=str)
    
    parser.add_argument("--explainer_mask_acti_func", default="relu", type=str)
    
    
    ## change these paths according to your environment
    parser.add_argument("--window_size", default=6, type=int, help="Size of the sliding window for data processing")
    parser.add_argument("--data_base_path", type=str, required=True, help="Base path for the dataset")
    parser.add_argument("--dataset_path", type=str, 
                        help="Path to the dataset file (auto-generated if not provided)")
    parser.add_argument("--model_path", type=str, 
                        help="Path to save/load the model checkpoint (auto-generated if not provided)")

    args = parser.parse_args()

    if args.dataset_path is None:
        args.dataset_path = f'{args.data_base_path}/temp_data/2022_CCF_AIOps_challenge/dataset/merge_multimodal/rca_multimodal_window_size_{args.window_size}.pkl'

    model_base_path = FileHandler.set_folder(f'{args.data_base_path}/model/a')
    checkpoint_dir = FileHandler.set_folder(model_base_path + "/checkpoint")

    # 如果 model_path 没有提供，则自动生成
    if args.model_path is None:
        args.model_path = f'{checkpoint_dir}/main.pt'
    ## end

    params = vars(parser.parse_args())

    rca_data_trainer = CRCATrainer(params)
    rca_data_trainer.train()
    rca_data_trainer.evaluate_rca_d3()
    ...
