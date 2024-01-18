import os
import time
import logging
import math
import copy
import argparse
import numpy as np
import pickle as plk
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

logger = logging.getLogger('MSA')

class MMERTD():
    def __init__(self, args):
        assert args.train_mode == 'regression'

        self.args = args
        self.args.tasks = "MTAV"
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)

        self.feature_map = {
            'fusion': torch.zeros(args.train_samples, args.post_fusion_dim, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, args.post_text_dim, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, args.post_audio_dim, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, args.post_video_dim, requires_grad=False).to(args.device),
        }

        self.center_map = {
            'fusion': {
                'pos': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
            },
            'text': {
                'pos': torch.zeros(args.post_text_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_text_dim, requires_grad=False).to(args.device),
            },
            'audio': {
                'pos': torch.zeros(args.post_audio_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_audio_dim, requires_grad=False).to(args.device),
            },
            'vision': {
                'pos': torch.zeros(args.post_video_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_video_dim, requires_grad=False).to(args.device),
            }
        }

        self.dim_map = {
            'fusion': torch.tensor(args.post_fusion_dim).float(),
            'text': torch.tensor(args.post_text_dim).float(),
            'audio': torch.tensor(args.post_audio_dim).float(),
            'vision': torch.tensor(args.post_video_dim).float(),
        }
        # new labels
        self.label_map = {
            'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, requires_grad=False).to(args.device)
        }

        self.name_map = {
            'M': 'fusion',
            'T': 'text',
            'A': 'audio',
            'V': 'vision'
        }

    def do_train(self, model, dataloader):
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.Model.text_model.named_parameters())
        audio_params = list(model.Model.audio_model.named_parameters())
        video_params = list(model.Model.video_model.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]
        model_params_other = [p for n, p in list(model.Model.named_parameters()) if 'text_model' not in n and \
                                'audio_model' not in n and 'video_model' not in n]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)

        saved_labels = {}
        # init labels
        logger.info("Init labels...")
        with tqdm(dataloader['train']) as td:
            for batch_data in td:
                labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                indexes = batch_data['index'].view(-1)
                self.init_labels(indexes, labels_m)

        # initilize results
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        M_pred_all, T_pred_all, A_pred_all, V_pred_all = [], [], [], []
        M_true_all, T_true_all, A_true_all = [], [], []
        f_vision_n , f_text_n , f_audio_n , f_fusion_n =[],[],[],[]
        # loop util earlystop
        while True:
            M_pred_all, T_pred_all, A_pred_all, V_pred_all = [], [], [], []
            M_true_all, T_true_all, A_true_all = [], [], []
            f_vision_n, f_text_n, f_audio_n, f_fusion_n = [], [], [], []
            epochs += 1
            # train
            y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': []}
            losses = []
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            ids = []
            # M_pred_all, T_pred_all, A_pred_all, V_pred_all = [], [], [], []
            # M_true_all, T_true_all, A_true_all = [], [], []
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    indexes = batch_data['index'].view(-1)
                    cur_id = batch_data['id']
                    ids.extend(cur_id)

                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    # forward
                    outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))
                    # store results
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(self.label_map[self.name_map[m]][indexes].cpu())
                    save_path = "saved_tsnee.npz"
                    M_pred_detached = [tensor.detach().numpy() for tensor in y_pred['M']]
                    T_pred_detached = [tensor.detach().numpy() for tensor in y_pred['T']]
                    A_pred_detached = [tensor.detach().numpy() for tensor in y_pred['A']]
                    V_pred_detached = [tensor.detach().numpy() for tensor in y_pred['V']]

                    M_true_detached = [tensor.detach().numpy() for tensor in y_true['M']]
                    T_true_detached = [tensor.detach().numpy() for tensor in y_true['T']]
                    A_true_detached = [tensor.detach().numpy() for tensor in y_true['A']]
                    # 将本次循环的结果保存到全局的数据结构中
                    M_pred_all.extend(M_pred_detached)
                    T_pred_all.extend(T_pred_detached)
                    A_pred_all.extend(A_pred_detached)
                    V_pred_all.extend(V_pred_detached)
                    M_true_all.extend(M_true_detached)
                    T_true_all.extend(T_true_detached)
                    A_true_all.extend(A_true_detached)
                    # np.savez(save_path,
                    #          M_pred=M_pred_detached,
                    #          T_pred=T_pred_detached,
                    #          A_pred=A_pred_detached,
                    #          V_pred=V_pred_detached,
                    #          M_true=M_true_detached,
                    #          T_true=T_true_detached,
                    #          A_true=A_true_detached,
                    #          )
                    np.savez(save_path,
                             M_pred=M_pred_all,
                             T_pred=T_pred_all,
                             A_pred=A_pred_all,
                             V_pred=V_pred_all,
                             M_true=M_true_all,
                             T_true=T_true_all,
                             A_true=A_true_all)

                    # compute loss
                    loss = 0.0
                    for m in self.args.tasks:
                        loss += self.weighted_loss(outputs[m], self.label_map[self.name_map[m]][indexes], \
                                                    indexes=indexes, mode=self.name_map[m])
                    # backward
                    loss.backward()
                    train_loss += loss.item()
                    # update features
                    f_fusion = outputs['Feature_f'].detach()
                    f_text = outputs['Feature_t'].detach()
                    f_audio = outputs['Feature_a'].detach()
                    f_vision = outputs['Feature_v'].detach()
                    f_fusion_n1 = f_fusion.cpu().numpy()
                    f_text_n1 = f_text.cpu().numpy()
                    f_audio_n1 = f_audio.cpu().numpy()
                    f_vision_n1 = f_vision.cpu().numpy()
                    f_fusion_n.extend(f_fusion_n1)
                    f_vision_n.extend(f_vision_n1)
                    f_text_n.extend(f_text_n1)
                    f_audio_n.extend(f_audio_n1)
                    M_true_detached = [tensor.detach().numpy() for tensor in y_true['M']]
                    flattened_array = np.concatenate([arr.ravel() for arr in M_true_detached])
                    save_path = "saved_features.npz"
                    np.savez(save_path,
                             fusion=f_fusion_n,
                             text=f_text_n,
                             audio=f_audio_n,
                             vision=f_vision_n,
                             labels = flattened_array,
                             epoch = epochs,)
                    if epochs > 1:
                        self.update_labels(f_fusion, f_text, f_audio, f_vision, epochs, indexes, outputs)

                    self.update_features(f_fusion, f_text, f_audio, f_vision, indexes)
                    self.update_centers()
                    
                    # update parameters
                    if not left_epochs:
                        # update
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:

                    # update
                    optimizer.step()
            train_loss = train_loss / len(dataloader['train'])
            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                        epochs-best_epoch, epochs, self.args.cur_time, train_loss))
            for m in self.args.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                train_results = self.metrics(pred, true)
                logger.info('%s: >> ' %(m) + dict_to_str(train_results))
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                #save result
                pred_best , true_best = torch.cat(y_pred['M']), torch.cat(y_true['M'])
                test_preds = pred_best.view(-1).cpu().detach().numpy()
                test_truth = true_best.view(-1).cpu().detach().numpy()
                test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
                test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
                test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
                test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
                test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
                test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)
                save_path = "saved_confusion.npz"
                np.savez(save_path,
                         test_preds=test_preds,
                         test_truth=test_truth,
                         test_preds_a7=test_preds_a7,
                         test_truth_a7=test_truth_a7,
                         test_preds_a5=test_preds_a5,
                         test_truth_a5=test_truth_a5,
                         test_preds_a3=test_preds_a3,
                         test_truth_a3=test_truth_a3, )
                """

                save_path = "saved_tsnee.npz"
                M_pred_detached = [tensor.detach().numpy() for tensor in y_pred['M']]
                T_pred_detached = [tensor.detach().numpy() for tensor in y_pred['T']]
                A_pred_detached = [tensor.detach().numpy() for tensor in y_pred['A']]
                V_pred_detached = [tensor.detach().numpy() for tensor in y_pred['V']]

                M_true_detached = [tensor.detach().numpy() for tensor in y_true['M']]
                T_true_detached = [tensor.detach().numpy() for tensor in y_true['T']]
                A_true_detached = [tensor.detach().numpy() for tensor in y_true['A']]
                np.savez(save_path,
                         M_pred=M_pred_detached,
                         T_pred=T_pred_detached,
                         A_pred=A_pred_detached,
                         V_pred=V_pred_detached,
                         M_true=M_true_detached,
                         T_true=T_true_detached,
                         A_true=A_true_detached,
                         )
                """

                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # save labels
            if self.args.save_labels:
                tmp_save = {k: v.cpu().numpy() for k, v in self.label_map.items()}
                tmp_save['ids'] = ids
                saved_labels[epochs] = tmp_save
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                if self.args.save_labels:
                    with open(os.path.join(self.args.res_save_dir, f'{self.args.modelName}-{self.args.datasetName}-labels.pkl'), 'wb') as df:
                        plk.dump(saved_labels, df, protocol=4)
                return

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        eval_loss = 0.0
        A_pred = []
        f_vision_nt, f_text_nt, f_audio_nt, f_fusion_nt = [], [], [], []

        # criterion = nn.L1Loss()
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    labels_m = batch_data['labels']['M'].to(self.args.device).view(-1)
                    outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))
                    loss = self.weighted_loss(outputs['M'], labels_m)
                    eval_loss += loss.item()
                    y_pred['M'].append(outputs['M'].cpu())
                    if mode =='TEST':
                        y_pred['A'].append(outputs['A'].cpu())
                        y_pred['T'].append(outputs['T'].cpu())
                        y_pred['V'].append(outputs['V'].cpu())
                        f_fusion = outputs['Feature_f'].detach()
                        f_text = outputs['Feature_t'].detach()
                        f_audio = outputs['Feature_a'].detach()
                        f_vision = outputs['Feature_v'].detach()
                        f_fusion_n1 = f_fusion.cpu().numpy()
                        f_text_n1 = f_text.cpu().numpy()
                        f_audio_n1 = f_audio.cpu().numpy()
                        f_vision_n1 = f_vision.cpu().numpy()
                        f_fusion_nt.extend(f_fusion_n1)
                        f_vision_nt.extend(f_vision_n1)
                        f_text_nt.extend(f_text_n1)
                        f_audio_nt.extend(f_audio_n1)
                    y_true['M'].append(labels_m.cpu())
                    M_true_detached = [tensor.detach().numpy() for tensor in y_true['M']]
                    flattened_array = np.concatenate([arr.ravel() for arr in M_true_detached])


        if mode == 'TEST':
            save_path = "test_features.npz"
            np.savez(save_path,
                     fusion=f_fusion_nt,
                     text=f_text_nt,
                     audio=f_audio_nt,
                     vision=f_vision_nt,
                     labels=flattened_array,
                     )
            flattened_tensors = [t.reshape(-1) for t in  y_pred['A']]  # 这会返回一个新的列表，其中每个元素都是一个一维数组
            result_A = np.concatenate(flattened_tensors)
            flattened_tensors = [t.reshape(-1) for t in  y_pred['T']]  # 这会返回一个新的列表，其中每个元素都是一个一维数组
            result_T = np.concatenate(flattened_tensors)
            flattened_tensors = [t.reshape(-1) for t in  y_pred['V']]  # 这会返回一个新的列表，其中每个元素都是一个一维数组
            result_V = np.concatenate(flattened_tensors)
            flattened_tensors = [t.reshape(-1) for t in  y_true['M']]  # 这会返回一个新的列表，其中每个元素都是一个一维数组
            result_M = np.concatenate(flattened_tensors)
            save_path = "test_tsne.npz"
            np.savez(save_path,
                     y_pred_A = result_A,
                     y_pred_T = result_T,
                     y_pred_V = result_V,
                     y_true_M = result_M,)
        pred_best, true_best = torch.cat(y_pred['M']), torch.cat(y_true['M'])
        test_preds = pred_best.view(-1).cpu().detach().numpy()
        test_truth = true_best.view(-1).cpu().detach().numpy()
        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
        test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)
        save_path = "saved_confusion_test.npz"
        np.savez(save_path,
                 test_preds=test_preds,
                 test_truth=test_truth,
                 test_preds_a7=test_preds_a7,
                 test_truth_a7=test_truth_a7,
                 test_preds_a5=test_preds_a5,
                 test_truth_a5=test_truth_a5,
                 test_preds_a3=test_preds_a3,
                 test_truth_a3=test_truth_a3, )
        # for m in self.args.tasks:
        #     y_pred[m].append(outputs[m].cpu())
        A_pred_detached = [tensor.detach().numpy() for tensor in y_pred['A']]
        A_pred.append(A_pred_detached)


        eval_loss = eval_loss / len(dataloader)
        logger.info(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        pred, true = torch.cat(y_pred['M']), torch.cat(y_true['M'])
        eval_results = self.metrics(pred, true)
        logger.info('M: >> ' + dict_to_str(eval_results))
        eval_results['Loss'] = eval_loss
        return eval_results
    
    def weighted_loss(self, y_pred, y_true, indexes=None, mode='fusion'):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        if mode == 'fusion':
            weighted = torch.ones_like(y_pred)
        else:
            weighted = torch.tanh(torch.abs(self.label_map[mode][indexes] - self.label_map['fusion'][indexes]))
        loss = torch.mean(weighted * torch.abs(y_pred - y_true))
        return loss
    
    def update_features(self, f_fusion, f_text, f_audio, f_vision, indexes):
        self.feature_map['fusion'][indexes] = f_fusion
        self.feature_map['text'][indexes] = f_text
        self.feature_map['audio'][indexes] = f_audio
        self.feature_map['vision'][indexes] = f_vision

    def update_centers(self):
        def update_single_center(mode):
            neg_indexes = self.label_map[mode] < 0
            if self.args.excludeZero:
                pos_indexes = self.label_map[mode] > 0
            else:
                pos_indexes = self.label_map[mode] >= 0
            self.center_map[mode]['pos'] = torch.mean(self.feature_map[mode][pos_indexes], dim=0)
            self.center_map[mode]['neg'] = torch.mean(self.feature_map[mode][neg_indexes], dim=0)

        update_single_center(mode='fusion')
        update_single_center(mode='text')
        update_single_center(mode='audio')
        update_single_center(mode='vision')
    
    def init_labels(self, indexes, m_labels):
        self.label_map['fusion'][indexes] = m_labels
        self.label_map['text'][indexes] = m_labels
        self.label_map['audio'][indexes] = m_labels
        self.label_map['vision'][indexes] = m_labels
    
    def update_labels(self, f_fusion, f_text, f_audio, f_vision, cur_epoches, indexes, outputs):
        MIN = 1e-8
        def update_single_label(f_single, mode):
            d_sp = torch.norm(f_single - self.center_map[mode]['pos'], dim=-1) 
            d_sn = torch.norm(f_single - self.center_map[mode]['neg'], dim=-1) 
            delta_s = (d_sn - d_sp) / (d_sp + MIN)
            # d_s_pn = torch.norm(self.center_map[mode]['pos'] - self.center_map[mode]['neg'], dim=-1)
            # delta_s = (d_sn - d_sp) / (d_s_pn + MIN)
            alpha = delta_s / (delta_f + MIN)

            new_labels = 0.5 * alpha * self.label_map['fusion'][indexes] + \
                        0.5 * (self.label_map['fusion'][indexes] + delta_s - delta_f)
            new_labels = torch.clamp(new_labels, min=-self.args.H, max=self.args.H)
            # new_labels = torch.tanh(new_labels) * self.args.H

            n = cur_epoches
            self.label_map[mode][indexes] = (n - 1) / (n + 1) * self.label_map[mode][indexes] + 2 / (n + 1) * new_labels

        d_fp = torch.norm(f_fusion - self.center_map['fusion']['pos'], dim=-1) 
        d_fn = torch.norm(f_fusion - self.center_map['fusion']['neg'], dim=-1) 
        # d_f_pn = torch.norm(self.center_map['fusion']['pos'] - self.center_map['fusion']['neg'], dim=-1)
        # delta_f = (d_fn - d_fp) / (d_f_pn + MIN)
        delta_f = (d_fn - d_fp) / (d_fp + MIN)
        
        update_single_label(f_text, mode='text')
        update_single_label(f_audio, mode='audio')
        update_single_label(f_vision, mode='vision')