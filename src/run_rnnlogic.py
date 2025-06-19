import sys
import os
import os.path as osp
import logging
import argparse
import random
import json
from easydict import EasyDict
import numpy as np
from datetime import datetime
import torch
# from torch.utils.data import DataLoader
from torch import distributed as dist

from data import KnowledgeGraph, TrainDataset, ValidDataset, TestDataset, RuleDataset
from predictors import Predictor, PredictorPlus
from generators import Generator
from generator_multitask import GeneratorMultitask
from utils import load_config, save_config, set_logger, set_seed, get_subset_dataset
from trainer import TrainerPredictor, TrainerGenerator
import comm

def save_rules(cfg, rules, label):
    relation_map = {}
    with open(cfg.data.data_path + '/relations.dict', 'r') as file:
        for line in file:
            rel_id, rel_name = line.strip().split("\t")
            relation_map[int(rel_id)] = rel_name
            
    all_readable_rules = []
    for rule in rules:
        readable_rule = []
        for rel in rule:
            readable_rule.append(relation_map.get(rel, f'UNKNOWN_REL_{rel}'))

        all_readable_rules.append(readable_rule)

    filepath = os.path.join(cfg.save_path, f'rules_{label}.txt')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(str(rule) for rule in all_readable_rules))

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='RNNLogic',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--config', type=str, default='../rnnlogic.yaml')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--mode', type=str, default='ori')
    parser.add_argument('--subset_ratio', type=float, default=1.0)
    parser.add_argument('--predictor_weighted_loss_mode', type=str, default='ori')
    parser.add_argument('--is_wrnnlogic', type=int, default=0, choices=[0, 1], help='Whether to use wRNNLogic (1) or RNNLogic (0).')

    return parser.parse_args(args)

def main(args):
    cfgs = load_config(args.config)
    cfg = cfgs[0]

    if cfg.save_path is None:
        cfg.save_path = os.path.join('../outputs', datetime.now().strftime('%Y%m-%d%H-%M%S'))
    
    if cfg.save_path and not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    
    save_config(cfg, cfg.save_path)

    set_logger(cfg.save_path)
    set_seed(cfg.seed)

    graph = KnowledgeGraph(cfg.data.data_path)
    train_set = TrainDataset(graph, cfg.data.batch_size)
    
    valid_set = ValidDataset(graph, cfg.data.batch_size)
    test_set = TestDataset(graph, cfg.data.batch_size)

    dataset = RuleDataset(graph.relation_size, cfg.data.rule_file, cfg.data.cluster_size, cfg.data.relation_cluster_file, args.is_wrnnlogic)


    if args.subset_ratio < 1.0:
        if comm.get_rank() == 0:
            logging.info('Using subset datasets for training, validation, and testing.')
            logging.info(f'Subset ratio: {args.subset_ratio}')
        train_set = get_subset_dataset(train_set, args.subset_ratio, seed=42)
        valid_set = get_subset_dataset(valid_set, args.subset_ratio, seed=99)   # 也可用不同 seed
        test_set  = get_subset_dataset(test_set,  args.subset_ratio, seed=123)
        dataset = get_subset_dataset(dataset, args.subset_ratio, seed=456)
    else:
        if comm.get_rank() == 0:
            logging.info('Using full datasets for training, validation, and testing.')



    if comm.get_rank() == 0:
        logging.info('-------------------------')
        logging.info('| Pre-train Generator')
        logging.info('-------------------------')
    
    if args.mode == 'ori':
        generator = Generator(graph, **cfg.generator.model)
    else:
        generator = GeneratorMultitask(graph, cfg.data.cluster_size, args.mode, **cfg.generator.model)
    solver_g = TrainerGenerator(generator, args.mode, gpu=cfg.generator.gpu)

    solver_g.train(dataset, **cfg.generator.pre_train)

    replay_buffer = list()
    for k in range(cfg.EM.num_iters):
        if comm.get_rank() == 0:
            logging.info('-------------------------')
            logging.info('| EM Iteration: {}/{}'.format(k + 1, cfg.EM.num_iters))
            logging.info('-------------------------')
        
        # Sample logic rules.
        sampled_rules = solver_g.sample(cfg.EM.num_rules, cfg.EM.max_length)
        prior = [rule[-1] for rule in sampled_rules]
        rules = [rule[0:-1] for rule in sampled_rules]
        save_rules(cfg, rules, k)

        # Train a reasoning predictor with sampled logic rules.
        predictor = Predictor(graph, args.is_wrnnlogic, **cfg.predictor.model)
        predictor.set_rules(rules)
        optim = torch.optim.Adam(predictor.parameters(), **cfg.predictor.optimizer)

        solver_p = TrainerPredictor(predictor, train_set, valid_set, test_set, optim, args.predictor_weighted_loss_mode, args.is_wrnnlogic, gpus=cfg.predictor.gpus)
        solver_p.train(**cfg.predictor.train)
        valid_mrr_iter = solver_p.evaluate('valid', expectation=cfg.predictor.eval.expectation)
        test_mrr_iter = solver_p.evaluate('test', expectation=cfg.predictor.eval.expectation)
        
        # E-step: Compute H scores of logic rules.
        likelihood = solver_p.compute_H(**cfg.predictor.H_score)
        posterior = [l + p * cfg.EM.prior_weight for l, p in zip(likelihood, prior)]
        for i in range(len(rules)):
            rules[i].append(posterior[i])
        replay_buffer += rules
        
        # M-step: Update the rule generator.
        dataset = RuleDataset(graph.relation_size, rules, cfg.data.cluster_size, cfg.data.relation_cluster_file)

        solver_g.train(dataset, **cfg.generator.train)
        
    if replay_buffer != []:
        if comm.get_rank() == 0:
            logging.info('-------------------------')
            logging.info('| Post-train Generator')
            logging.info('-------------------------')
        dataset = RuleDataset(graph.relation_size, replay_buffer, cfg.data.cluster_size, cfg.data.relation_cluster_file)
        solver_g.train(dataset, **cfg.generator.post_train)

    if comm.get_rank() == 0:
        logging.info('-------------------------')
        logging.info('| Beam Search Best Rules')
        logging.info('-------------------------')
    
    sampled_rules = list()
    for num_rules, max_length in zip(cfg.final_prediction.num_rules, cfg.final_prediction.max_length):
        sampled_rules_ = solver_g.beam_search(num_rules, max_length)
        sampled_rules += sampled_rules_
        
    prior = [rule[-1] for rule in sampled_rules]
    rules = [rule[0:-1] for rule in sampled_rules]

    if args.subset_ratio < 1.0:
        if comm.get_rank() == 0:
            logging.info('Using subset datasets for final predictor+ training, validation, and testing.')
            logging.info(f'Subset ratio: {args.subset_ratio}')

            subset_size = int(len(rules) * args.subset_ratio)
            subset_indices = random.sample(range(len(rules)), subset_size)

            rules = [rules[i] for i in subset_indices]
            prior = [prior[i] for i in subset_indices]

            logging.info(f'Subset sizes: train={len(train_set)}, valid={len(valid_set)}, test={len(test_set)}, rule dataset={len(rules)}')


    if comm.get_rank() == 0:
        logging.info('-------------------------')
        logging.info('| Train Final Predictor+')
        logging.info('-------------------------')

    predictor = PredictorPlus(graph, **cfg.predictorplus.model)
    predictor.set_rules(rules)
    save_rules(cfg, rules, "final")

    optim = torch.optim.Adam(predictor.parameters(), **cfg.predictorplus.optimizer)

    solver_p = TrainerPredictor(predictor, train_set, valid_set, test_set, optim, args.predictor_weighted_loss_mode, args.is_wrnnlogic, gpus=cfg.predictorplus.gpus)
    best_valid_mrr = 0.0
    test_mrr = 0.0
    for k in range(cfg.final_prediction.num_iters):
        if comm.get_rank() == 0:
            logging.info('-------------------------')
            logging.info('| Iteration: {}/{}'.format(k + 1, cfg.final_prediction.num_iters))
            logging.info('-------------------------')

        solver_p.train(**cfg.predictorplus.train)
        valid_mrr_iter = solver_p.evaluate('valid', expectation=cfg.predictorplus.eval.expectation)
        test_mrr_iter = solver_p.evaluate('test', expectation=cfg.predictorplus.eval.expectation)

        if valid_mrr_iter > best_valid_mrr:
            best_valid_mrr = valid_mrr_iter
            test_mrr = test_mrr_iter
            solver_p.save(os.path.join(cfg.save_path, 'predictor.pt'))
    
    if comm.get_rank() == 0:
        logging.info('-------------------------')
        logging.info('| Final Test MRR: {:.6f}'.format(test_mrr))
        logging.info('-------------------------')

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main(parse_args())