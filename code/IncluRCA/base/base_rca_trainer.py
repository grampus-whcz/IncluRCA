from abc import ABC
from shared_util.logger import Logger
import torch
from IncluRCA.data_loader.rca_data_loader import RCADataLoader
from IncluRCA.model.o11y.representation_learning import RepresentationLearning
# from IncluRCA.model.re.feature_integration import FeatureIntegration
# from IncluRCA.model.re.feature_integration_MultiScaleConvSEAttention import FeatureIntegration
from IncluRCA.model.re.feature_integration_SEAttention import FeatureIntegration
# from IncluRCA.model.re.feature_integration_ECAAttention import FeatureIntegration
# from IncluRCA.model.re.feature_integration_SKAttention1D import FeatureIntegration
# from IncluRCA.model.re.feature_integration_TripletAttention1D import FeatureIntegration
# from IncluRCA.model.re.feature_integration_TASAttention import FeatureIntegration # Local optimal
# from IncluRCA.model.re.feature_integration_TAAttention import FeatureIntegration
# from IncluRCA.model.re.feature_integration_CTMSA import FeatureIntegration
from IncluRCA.model.re.feature_fusion import FeatureFusion
# from IncluRCA.model.re.feature_fusion_new import FeatureFusion
from IncluRCA.model.re.fault_classifier import FaultClassifier
from IncluRCA.util.data_handler import rearrange_y
from shared_util.evaluation_metrics import *
from torchinfo import summary


class BaseRCATrainer(ABC):
    def __init__(self, param_dict):
        self.param_dict = param_dict
        assert param_dict['orl_te_in_channels'] == param_dict['efi_in_dim'] and param_dict['efi_out_dim'] == param_dict['eff_in_dim']
        self.device = torch.device("cuda" if torch.cuda.is_available() and param_dict['gpu'] else "cpu")

        self.logger = Logger(logging_level='DEBUG').logger

        self.rca_data_loader = RCADataLoader(param_dict)
        self.rca_data_loader.load_data(f'{self.param_dict["dataset_path"]}')

        o11y_representation_learning = RepresentationLearning(param_dict, self.rca_data_loader.meta_data)
        re_feature_integration = FeatureIntegration(param_dict, self.rca_data_loader.meta_data)
        re_feature_fusion = FeatureFusion(param_dict, self.rca_data_loader.meta_data)
        re_fault_classifier = FaultClassifier(param_dict, self.rca_data_loader.meta_data)
        self.model = torch.nn.Sequential(o11y_representation_learning, re_feature_integration, re_feature_fusion, re_fault_classifier).to(self.device)
        self.model_rank = []
        
        summary(self.model)

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param_dict['lr'], weight_decay=self.param_dict['weight_decay'])

        criterion_dict = dict()
        for ent_type in self.rca_data_loader.meta_data['ent_types']:
            pos_weight = torch.FloatTensor(self.rca_data_loader.meta_data['ent_fault_type_weight'][ent_type]).to(self.device)
            criterion_dict[ent_type] = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(self.param_dict['epochs']):
            self.model.train()
            train_loss = 0
            for batch_id, batch_data in enumerate(self.rca_data_loader.data_loader['train']):
                optimizer.zero_grad()
                y = rearrange_y(self.rca_data_loader.meta_data, batch_data['y'], self.device)
                out = self.model(batch_data)
                loss = 0
                for ent_type in self.rca_data_loader.meta_data['ent_types']:
                    loss += criterion_dict[ent_type](out[ent_type][torch.where(~(y[ent_type] == -1).all(1))[0]], y[ent_type][torch.where(~(y[ent_type] == -1).all(1))[0]])
                train_loss += batch_data['y'].shape[0] * loss.item()
                loss.backward()
                optimizer.step()
            train_loss /= len(self.rca_data_loader.data_loader['train'].dataset)
            self.logger.info(f'[{epoch}/{self.param_dict["epochs"]}] | train_loss: {train_loss:.5f}')

            self.model.eval()
            y_pred = dict()
            y_true = dict()
            if epoch % 10 == 0:
                with torch.no_grad():
                    for batch_id, batch_data in enumerate(self.rca_data_loader.data_loader['valid']):
                        y = rearrange_y(self.rca_data_loader.meta_data, batch_data['y'], self.device)
                        out = self.model(batch_data)
                        for ent_type in self.rca_data_loader.meta_data['ent_types']:
                            if ent_type not in y_pred.keys():
                                y_pred[ent_type] = []
                                y_true[ent_type] = []
                            y_pred[ent_type].extend((torch.sigmoid(out[ent_type][y[ent_type] != -1].reshape(-1, out[ent_type].shape[1])) > self.param_dict[f'{ent_type}_accuracy_th']).cpu().detach().numpy())
                            y_true[ent_type].extend(y[ent_type][y[ent_type] != -1].reshape(-1, y[ent_type].shape[1]).cpu().detach().numpy())
                self.output_evaluation_rca_d3_result(y_pred, y_true, 'valid')
        torch.save(self.model.state_dict(), self.param_dict["model_path"])

    def evaluate_rca_d3(self):
        self.model.eval()
        self.model.load_state_dict(torch.load(self.param_dict["model_path"]))

        y_pred, y_true = dict(), dict()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(self.rca_data_loader.data_loader['test']):
                y = rearrange_y(self.rca_data_loader.meta_data, batch_data['y'], self.device)
                out = self.model(batch_data)
                for ent_type in self.rca_data_loader.meta_data['ent_types']:
                    fault_prob = torch.sigmoid(out[ent_type])
                    temp_y_pred = (fault_prob > self.param_dict[f'{ent_type}_accuracy_th']).cpu().detach().numpy()
                    temp_y_true = y[ent_type].cpu().detach().numpy()
                    if ent_type not in y_pred.keys():
                        y_pred[ent_type] = []
                        y_true[ent_type] = []
                    y_pred[ent_type].extend(temp_y_pred)
                    y_true[ent_type].extend(temp_y_true)

        self.output_evaluation_rca_d3_result(y_pred, y_true, 'test')

    def output_evaluation_rca_d3_result(self, y_pred, y_true, dataset_type):
        self.logger.info('----------')
        self.logger.info(f'evaluation dataset type: {dataset_type}')
        for ent_type in self.rca_data_loader.meta_data['ent_types']:
            ent_y_pred = np.array(y_pred[ent_type])
            ent_y_true = np.array(y_true[ent_type])
            fc_result = fault_type_classification(ent_y_pred, ent_y_true)
            convert = {
                'p': 'precision',
                'r': 'recall',
                'f1': 'f1'
            }
            for em in ['p', 'r', 'f1']:
                self.logger.info(f'{ent_type.ljust(8) + convert[em].ljust(9)} | micro: {fc_result["micro_" + convert[em] + "_score"]:.6f}; macro: {fc_result["macro_" + convert[em] + "_score"]:.6f}; score: {fc_result[convert[em] + "_score"]}')
        self.logger.info('----------')
        
    def diagnose_prediction_errors(self):
        import numpy as np
        from collections import Counter

        self.model.eval()
        self.model.load_state_dict(torch.load(self.param_dict["model_path"]))

        y_pred_list = dict()
        y_true_list = dict()

        with torch.no_grad():
            for batch_id, batch_data in enumerate(self.rca_data_loader.data_loader['test']):
                y = rearrange_y(self.rca_data_loader.meta_data, batch_data['y'], self.device)
                out = self.model(batch_data)

                for ent_type in self.rca_data_loader.meta_data['ent_types']:
                    fault_prob = torch.sigmoid(out[ent_type])
                    if ent_type not in y_pred_list:
                        y_pred_list[ent_type] = []
                        y_true_list[ent_type] = []
                    y_pred_list[ent_type].append(fault_prob.cpu())
                    y_true_list[ent_type].append(y[ent_type].cpu())

        # 合并
        y_pred_cat = {}
        y_true_cat = {}
        for ent_type in self.rca_data_loader.meta_data['ent_types']:
            y_pred_cat[ent_type] = torch.cat(y_pred_list[ent_type], dim=0)
            y_true_cat[ent_type] = torch.cat(y_true_list[ent_type], dim=0)

        total_node_entities = y_pred_cat['node'].shape[0]
        N_cases = total_node_entities // 6
        assert total_node_entities % 6 == 0
        assert y_pred_cat['service'].shape[0] == N_cases * 10
        assert y_pred_cat['pod'].shape[0] == N_cases * 40

        y_pred_case = {
            'node': y_pred_cat['node'].view(N_cases, 6, -1),
            'service': y_pred_cat['service'].view(N_cases, 10, -1),
            'pod': y_pred_cat['pod'].view(N_cases, 40, -1),
        }
        y_true_case = {
            'node': y_true_cat['node'].view(N_cases, 6, -1),
            'service': y_true_cat['service'].view(N_cases, 10, -1),
            'pod': y_true_cat['pod'].view(N_cases, 40, -1),
        }

        th = {
            'node': self.param_dict['node_accuracy_th'],
            'service': self.param_dict['service_accuracy_th'],
            'pod': self.param_dict['pod_accuracy_th'],
        }

        # 辅助函数：判断一个 case 的真实故障类型
        def get_fault_type(case_idx):
            for ent_type in ['node', 'service', 'pod']:
                true_labels = y_true_case[ent_type][case_idx]  # (E, C)
                # 只考虑有效标签（!= -1）
                valid_mask = (true_labels != -1)
                if valid_mask.any():
                    # 检查是否有任何位置为 1
                    if (true_labels[valid_mask] == 1).any():
                        return ent_type
            return "unknown"

        correct_cases = 0
        error_cases_details = []
        correct_cases_details = []

        for case_idx in range(N_cases):
            fault_type = get_fault_type(case_idx)

            case_is_correct = True
            errors_in_case = []
            entities_in_case = []

            for ent_type, num_entities in [('node', 6), ('service', 10), ('pod', 40)]:
                pred_probs = y_pred_case[ent_type][case_idx]
                true_labels = y_true_case[ent_type][case_idx]
                pred_binary = (pred_probs > th[ent_type]).float()

                for ent_id in range(num_entities):
                    true_vec = true_labels[ent_id]
                    pred_vec = pred_binary[ent_id]
                    valid_mask = (true_vec != -1)

                    if not valid_mask.any():
                        continue

                    true_valid = true_vec[valid_mask].cpu().numpy()
                    pred_valid = pred_vec[valid_mask].cpu().numpy()
                    prob_valid = pred_probs[ent_id][valid_mask].cpu().numpy()

                    entity_info = {
                        'entity_type': ent_type,
                        'entity_id': ent_id,
                        'true': true_valid.tolist(),
                        'pred': pred_valid.tolist(),
                        'prob': prob_valid.tolist()
                    }

                    if not np.array_equal(true_valid, pred_valid):
                        case_is_correct = False
                        errors_in_case.append(entity_info)

                    entities_in_case.append(entity_info)

            record = {
                'test_case_id': case_idx,
                'fault_type': fault_type,
                'entities': entities_in_case
            }

            if case_is_correct:
                correct_cases += 1
                correct_cases_details.append(record)
            else:
                error_cases_details.append({
                    'test_case_id': case_idx,
                    'fault_type': fault_type,
                    'errors': errors_in_case
                })

        self.logger.info("========== PREDICTION DIAGNOSIS ==========")
        self.logger.info(f"Total test cases: {N_cases}")
        self.logger.info(f"Correctly predicted cases: {correct_cases}")
        self.logger.info(f"Error cases: {len(error_cases_details)}")
        self.logger.info(f"Case-level accuracy: {correct_cases / N_cases:.4f}")

        # ===== 打印所有正确 case（带故障类型）=====
        self.logger.info("\n--- ALL Correctly Predicted Cases ---")
        for corr in correct_cases_details:
            self.logger.info(f"\n--- Correct Case (Test Case ID: {corr['test_case_id']}) | Fault Type: {corr['fault_type']} ---")
            # for e in corr['entities']:
            #     self.logger.info(f"  {e['entity_type']}[{e['entity_id']}]: true={e['true']}, pred={e['pred']}, prob={e['prob']}")

        # ===== 打印所有错误 case（带故障类型）=====
        self.logger.info("\n--- ALL Error Cases ---")
        for err in error_cases_details:
            self.logger.info(f"\n--- Error Case (Test Case ID: {err['test_case_id']}) | Fault Type: {err['fault_type']} ---")
            for e in err['errors']:
                self.logger.info(f"  {e['entity_type']}[{e['entity_id']}]: true={e['true']}, pred={e['pred']}, prob={e['prob']}")

        # ===== 错误类型统计 =====
        fault_error_counter = Counter()
        for err in error_cases_details:
            for e in err['errors']:
                true_arr = np.array(e['true'])
                pred_arr = np.array(e['pred'])
                if len(true_arr) != len(pred_arr):
                    continue
                diff_idx = np.where(true_arr != pred_arr)[0]
                for idx in diff_idx:
                    if true_arr[idx] == 1:
                        fault_error_counter[(e['entity_type'], idx, 'FN')] += 1
                    else:
                        fault_error_counter[(e['entity_type'], idx, 'FP')] += 1

        self.logger.info("\nTop fault-type errors:")
        for (ent, ftype, err_type), cnt in fault_error_counter.most_common(10):
            self.logger.info(f"  {ent} fault_type[{ftype}] {err_type}: {cnt} times")

        self.logger.info("==========================================")
