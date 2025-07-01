import os
import logging
import argparse
import random
from datetime import datetime
import torch
import torch.distributed as dist

from data.graph import KnowledgeGraph
from data.triples_dataset import TrainDataset, ValidDataset, TestDataset
from data.rules_dataset import RuleDataset

from predictors import Predictor, PredictorPlus
from generators import Generator

from generator_multitask_models.multitask_mmoe import MultitaskMMOE
from generator_multitask_models.multitask_hard_sharing import MultitaskHardSharing

from utils import load_config, save_config, set_logger, set_seed, get_subset_dataset
from trainer import TrainerPredictor, TrainerGenerator
import comm
from grd_for_aux import grd2encoding

# ------- util -----------
def bcast_obj(obj, src=0):
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size()==1:
        return obj
    buf = [obj] if dist.get_rank()==src else [None]
    dist.broadcast_object_list(buf, src)
    return buf[0]

def str2bool(v): # args parsing
    return str(v).lower() in ("True", "true", "t", "1")

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
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--config', type=str, default='../rnnlogic.yaml')
    parser.add_argument('--subset_ratio', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='ori')
    parser.add_argument('--multitask_loss_mode', type=str, default='adaptive')
    parser.add_argument('--predictor_weighted_loss_mode', type=str, default='ori')
    parser.add_argument('--is_soft_label', type=str2bool, default=False)
    parser.add_argument('--is_scheduled_sampling', type=str2bool, default=False)
    parser.add_argument('--type_or_group', type=str, default='type')
    parser.add_argument('--is_test_mode', type=str2bool, default=False)

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
    logging.info(f"is_soft_label: {args.is_soft_label}")
    logging.info(f"is_scheduled_sampling: {args.is_scheduled_sampling}")
    logging.info(f"is_test_mode: {args.is_test_mode}")

    graph = KnowledgeGraph(cfg.data.data_path, args.type_or_group)
    train_set = TrainDataset(graph, cfg.data.batch_size)
    
    valid_set = ValidDataset(graph, cfg.data.batch_size)
    test_set = TestDataset(graph, cfg.data.batch_size)

    dataset = RuleDataset(graph.relation_size, cfg.data.rule_file, cfg.data.cluster_size, cfg.data.relation_cluster_file)


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

    # Init generator
    if args.model == 'ori':
        generator = Generator(graph, **cfg.generator.model)
    elif args.model == 'multitask_hard_sharing':
        generator = MultitaskHardSharing(graph, cfg.data.cluster_size, **cfg.generator.model)
    else: # multitask_mmoe
        generator = MultitaskMMOE(graph, cfg.data.cluster_size, is_soft_label=args.is_soft_label, **cfg.generator.model)

    solver_g = TrainerGenerator(generator, args.model, args.multitask_loss_mode, args.is_scheduled_sampling, gpu=cfg.generator.gpu)

    replay_buffer = list()
    for k in range(cfg.EM.num_iters):
        if comm.get_rank() == 0:
            logging.info('-------------------------')
            logging.info('| EM Iteration: {}/{}'.format(k + 1, cfg.EM.num_iters))
            logging.info('-------------------------')

        if k == 0:
            # EM round 0: Use mined rules as RP's input
            logging.info('>>>>> Using miner\'s rule')
            sampled_rules = dataset.rp_input
            if args.is_test_mode:
                sampled_rules = sampled_rules[:10] 
        else:
            # Sample logic rules with Generator
            logging.info('>>>>> Using Generator\'s sample rule')
            sampled_rules = solver_g.sample(cfg.EM.num_rules, cfg.EM.max_length)

        prior = [rule[-1] for rule in sampled_rules]
        rules = [rule[0:-1] for rule in sampled_rules]
        save_rules(cfg, rules, k) # 記錄用

        predictor = Predictor(graph, **cfg.predictor.model)
        predictor.set_rules(rules)
        optim = torch.optim.Adam(predictor.parameters(), **cfg.predictor.optimizer)

        solver_p = TrainerPredictor(predictor, train_set, valid_set, test_set, optim, args.predictor_weighted_loss_mode, gpus=cfg.predictor.gpus)
        solver_p.train(**cfg.predictor.train)
        valid_mrr_iter = solver_p.evaluate('valid', expectation=cfg.predictor.eval.expectation)
        test_mrr_iter = solver_p.evaluate('test', expectation=cfg.predictor.eval.expectation)

        # E-step: Compute H scores of logic rules.
        dump_dir = "grd_dumps"
        dump_path = os.path.join(
            cfg.save_path,
            dump_dir,              # 子資料夾
            f"edge_iter_{k}.jsonl.gz"   # 檔名
        )
        # ── 建資料夾（不存在就自動建立）────────────────────────
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)

        likelihood = solver_p.compute_H(dump_path, **cfg.predictor.H_score) # 順便算 groundings
        posterior = [l + p * cfg.EM.prior_weight for l, p in zip(likelihood, prior)]
        for i in range(len(rules)):
            rules[i].append(posterior[i])
        replay_buffer += rules
        
        # M-step: Update the rule generator.
        grd_encoding = grd2encoding(rules, dump_path, graph.ent2types, len(graph.id2type), is_soft_label=args.is_soft_label) # old: graph.ent2type
        assert len(grd_encoding) == len(rules)
        for mh, r in zip(grd_encoding, rules):
            # mh.shape[0] 應該 ＝ len(r)   (rule_head + body)   ← 先不含 END
            assert mh.shape[0] == len(r), f"{mh.shape} vs rule len {len(r)}"
            assert mh.shape[1] == len(graph.id2type)

        dataset = RuleDataset(graph.relation_size, rules, cfg.data.cluster_size, cfg.data.relation_cluster_file)
        dataset.update_grd_multihot(grd_encoding, len(graph.id2type))

        solver_g.train(dataset, **cfg.generator.train)

    if replay_buffer != []:
        if comm.get_rank() == 0:
            logging.info('-------------------------')
            logging.info('| Post-train Generator')
            logging.info('-------------------------')
        # grd_encoding = grd2encoding(replay_buffer, dump_path, graph.ent2type, len(graph.id2type), is_soft_label=SOFT_LABEL) # rule_num * rule_len + 1(head) + 1(end) * 
        grd_encoding = grd2encoding(replay_buffer, dump_path, graph.ent2types, len(graph.id2type), is_soft_label=args.is_soft_label) # rule_num * rule_len + 1(head) + 1(end) * 
        
        dataset = RuleDataset(graph.relation_size, replay_buffer, cfg.data.cluster_size, cfg.data.relation_cluster_file)
        dataset.update_grd_multihot(grd_encoding, len(graph.id2type))
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
    rules = bcast_obj(rules, src=0)   

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

    solver_p = TrainerPredictor(predictor, train_set, valid_set, test_set, optim, args.predictor_weighted_loss_mode, gpus=cfg.predictorplus.gpus)
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