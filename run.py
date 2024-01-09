import os
import gc
import time
import random
import torch
import pynvml
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader
from config.config_regression import ConfigRegression

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_experiment(args, seed, iteration):
    if args.train_mode == "regression":
        config = ConfigRegression(args)
        args = config.get_config()
    setup_seed(seed)
    args.seed = seed
    logger.info(f'Start running {args.modelName}...')
    logger.info(args)
    args.cur_time = iteration + 1
    return args


def run(args):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    args.model_save_path = os.path.join(args.model_save_dir,\
                                        f'{args.modelName}-{args.datasetName}-{args.train_mode}.pth')
    
    if len(args.gpu_ids) == 0 and torch.cuda.is_available():
        # load free-most gpu
        pynvml.nvmlInit()
        dst_gpu_id, min_mem_used = 0, 1e16
        for g_id in [0, 1, 2, 3]:
            handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        logger.info(f'Find gpu: {dst_gpu_id}, with memory: {min_mem_used} left!')
        args.gpu_ids.append(dst_gpu_id)
    # device
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    logger.info("Let's use %d GPUs!" % len(args.gpu_ids))
    # device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')
    device = torch.device('cuda:0'  if using_cuda else 'cpu')
    args.device = device
    # data
    dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)

    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()

        return answer
    logger.info(f'The model has {count_parameters(model)} trainable parameters')

    atio = ATIO().getTrain(args)
    # do train
    atio.do_train(model, dataloader)
    # load pretrained model
    assert os.path.exists(args.model_save_path)
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(device)

    # do test
    if args.tune_mode:
        # using valid dataset to debug hyper parameters
        results = atio.do_test(model, dataloader['valid'], mode="VALID")
    else:
        results = atio.do_test(model, dataloader['test'], mode="TEST")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results


def run_normal(args):
    setup_results_directory(args)
    model_results = execute_models(args)
    save_model_results(args, model_results)

def setup_results_directory(args):
    args.res_save_dir = os.path.join(args.res_save_dir, 'normals')
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)

def execute_models(args):
    model_results = []
    for i, seed in enumerate(args.seeds):
        args = setup_experiment(args, seed, i)
        test_results = run(args)
        model_results.append(test_results)
    return model_results

def set_log(args):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_file_path = f'logs/{args.modelName}-{args.datasetName}.log'
    # set logging
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)
    return logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', type=str, default="regression",
                        help='regression / classification')
    parser.add_argument('--modelName', type=str, default='MMERTD',
                        help='support MMERTD')
    parser.add_argument('--datasetName', type=str, default='mosi',
                        help='support mosi/mosei/sims')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default='results/models',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/results',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[1],
                        help='indicates the gpus will be used. If none, the most-free gpu will be used!')
    return parser.parse_args()


def save_model_results(args, model_results):
    criterions = list(model_results[0].keys())
    save_path = os.path.join(args.res_save_dir, f'{args.datasetName}-{args.train_mode}.csv')
    df = pd.read_csv(save_path) if os.path.exists(save_path) else pd.DataFrame(columns=["Model"] + criterions)
    df.loc[len(df)] = compute_results(args, model_results, criterions)
    df.to_csv(save_path, index=False)
    logger.info(f'Results are added to {save_path}...')

def compute_results(args, model_results, criterions):
    res = [args.modelName]
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values) * 100, 2)
        std = round(np.std(values) * 100, 2)
        res.append((mean, std))
    return res



if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()

    for data_name in ['mosi']:
        args.datasetName = data_name
        logger = set_log(args)

        args.seeds = [1111,1112,1115] 


        run_normal(args)
    # 记录程序结束时间
    end_time = time.time()

    # 计算运行时间
    run_time = end_time - start_time

    # 打印程序运行时间
    print(f"程序运行时间：{run_time} 秒")