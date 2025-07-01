import comm
from utils import *
import torch
from torch import distributed as dist
from torch import nn
from torch.utils import data as torch_data
from itertools import islice
from data.rules_dataset import RuleDataset
from datamodules.utils import Iterator
import torch.nn.functional as F
import gzip, json

class TrainerPredictor(object):

    def __init__(self, model, train_set, valid_set, test_set, optimizer, weighted_loss_mode, scheduler=None, gpus=None, num_worker=0):
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.gpus = gpus
        self.num_worker = num_worker
        self.weighted_loss_mode = weighted_loss_mode

        if gpus is None:
            self.device = torch.device("cpu")
        else:
            if len(gpus) != self.world_size:
                error_msg = "World size is %d but found %d GPUs in the argument"
                if self.world_size == 1:
                    error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
                raise ValueError(error_msg % (self.world_size, len(gpus)))
            self.device = torch.device(gpus[self.rank % len(gpus)])

        if self.world_size > 1 and not dist.is_initialized():
            if self.rank == 0:
                logging.info("Initializing distributed process group")
            backend = "gloo" if gpus is None else "nccl"
            comm.init_process_group(backend, init_method="env://")

        if self.rank == 0:
            logging.info("Preprocess training set")
        if self.world_size > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if self.device.type == "cuda":
            model = model.cuda(self.device)

        self.model = model
        self.train_set = train_set
        
        self.valid_set = valid_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, batch_per_epoch, smoothing, print_every):
        if comm.get_rank() == 0:
            logging.info('>>>>> Predictor: Training')
        self.train_set.make_batches()

        sampler = torch_data.DistributedSampler(self.train_set, self.world_size, self.rank)
        dataloader = torch_data.DataLoader(self.train_set, 1, sampler=sampler, num_workers=self.num_worker)
        batch_per_epoch = batch_per_epoch or len(dataloader)
        model = self.model
        if self.world_size > 1:
            if self.device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device], find_unused_parameters=True)
            else:
                model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.train()

        total_loss = 0.0
        total_size = 0.0

        sampler.set_epoch(0)

        for batch_id, batch in enumerate(islice(dataloader, batch_per_epoch)):
            all_h, all_r, all_t, target, edges_to_remove, triple_weight = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            target = target.squeeze(0)
            edges_to_remove = edges_to_remove.squeeze(0)
            target_t = torch.nn.functional.one_hot(all_t, self.train_set.graph.entity_size)

            triple_weight = triple_weight.squeeze(0)
            
            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                target = target.cuda(device=self.device)
                edges_to_remove = edges_to_remove.cuda(device=self.device)
                target_t = target_t.cuda(device=self.device)
                triple_weight = triple_weight.cuda(device=self.device)

            
            target = target * smoothing + target_t * (1 - smoothing)
            logits, mask = model(all_h, all_r, edges_to_remove) # 這邊的 logits 是 score

            if mask.sum().item() != 0:
                logits = (torch.softmax(logits, dim=1) + 1e-8).log()

                if self.weighted_loss_mode == 'ori':
                    loss = -(logits[mask] * target[mask]).sum() / torch.clamp(target[mask].sum(), min=1)
                else:
                    if self.weighted_loss_mode == 'triple_count':
                        weighted_log = logits * triple_weight 
                    elif self.weighted_loss_mode == 'triple_count_sqrt':
                        weighted_log = logits * triple_weight.sqrt()
                    elif self.weighted_loss_mode == 'triple_count_log':
                        weighted_log = logits * torch.log1p(triple_weight)

                    loss = -(weighted_log[mask] * target[mask]).sum() / torch.clamp(target[mask].sum(), min=1)

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                total_size += mask.sum().item()
                
            if (batch_id + 1) % print_every == 0:
                if comm.get_rank() == 0:
                    logging.info('{} {} {:.6f} {:.1f}'.format(batch_id + 1, len(dataloader), total_loss / print_every, total_size / print_every))
                total_loss = 0.0
                total_size = 0.0

        if self.scheduler:
            self.scheduler.step()
    
    @torch.no_grad()
    def compute_H(self, dump_path=None, print_every=1000):
        if comm.get_rank() == 0:
            logging.info('>>>>> Predictor: Computing H scores of rules')
        sampler = torch_data.DistributedSampler(self.train_set, self.world_size, self.rank)
        dataloader = torch_data.DataLoader(self.train_set, 1, sampler=sampler, num_workers=self.num_worker)
        model = self.model

        model.eval()
        all_H_score = torch.zeros(model.num_rules, device=self.device)

        record_pool = [] if dump_path is not None else None   # ★ ①

        for batch_id, batch in enumerate(dataloader):
            all_h, all_r, all_t, target, edges_to_remove, triple_weight = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            target = target.squeeze(0)
            edges_to_remove = edges_to_remove.squeeze(0)
            
            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                target = target.cuda(device=self.device)
                edges_to_remove = edges_to_remove.cuda(device=self.device)
            
            H, index = model.compute_H(
                all_h, all_r, all_t, edges_to_remove,
                record_pool=record_pool                      # ★ ②
            )
            if H != None and index != None:
                all_H_score[index] += H / len(model.graph.train_facts)
                
            if (batch_id + 1) % print_every == 0:
                if comm.get_rank() == 0:
                    logging.info('{} {}'.format(batch_id + 1, len(dataloader)))
        
        if self.world_size > 1:
            all_H_score = comm.stack(all_H_score)
            all_H_score = all_H_score.sum(0)

        # logging.info(f"record pool: {record_pool}")

        # 需要保存路徑時寫檔
        if record_pool is not None and comm.get_rank() == 0:   # ★ ③
            with gzip.open(dump_path, "wt") as fp:
                for rec in record_pool:
                    fp.write(json.dumps(rec) + "\n")
            record_pool.clear()

        
        return all_H_score.data.cpu().numpy().tolist()
    
    @torch.no_grad()
    def evaluate(self, split, expectation=True):
        if comm.get_rank() == 0:
            logging.info('>>>>> Predictor: Evaluating on {}'.format(split))
        test_set = getattr(self, "%s_set" % split)
        sampler = torch_data.DistributedSampler(test_set, self.world_size, self.rank)
        dataloader = torch_data.DataLoader(test_set, 1, sampler=sampler, num_workers=self.num_worker)
        model = self.model

        model.eval()
        concat_logits = []
        concat_all_h = []
        concat_all_r = []
        concat_all_t = []
        concat_flag = []
        concat_mask = []
        for batch in dataloader:
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            flag = flag.squeeze(0)
            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                all_t = all_t.cuda(device=self.device)
                flag = flag.cuda(device=self.device)

            # TODO: original
            logits, mask = model(all_h, all_r, None)
        
            concat_logits.append(logits)
            concat_all_h.append(all_h)
            concat_all_r.append(all_r)
            concat_all_t.append(all_t)
            concat_flag.append(flag)
            concat_mask.append(mask)
        
        concat_logits = torch.cat(concat_logits, dim=0)
        concat_all_h = torch.cat(concat_all_h, dim=0)
        concat_all_r = torch.cat(concat_all_r, dim=0)
        concat_all_t = torch.cat(concat_all_t, dim=0)
        concat_flag = torch.cat(concat_flag, dim=0)
        concat_mask = torch.cat(concat_mask, dim=0)
        
        ranks = []
        for k in range(concat_all_t.size(0)):
            h = concat_all_h[k]
            r = concat_all_r[k]
            t = concat_all_t[k]
            if concat_mask[k, t].item() == True:
                val = concat_logits[k, t]
                L = (concat_logits[k][concat_flag[k]] > val).sum().item() + 1
                H = (concat_logits[k][concat_flag[k]] >= val).sum().item() + 2
            else:
                L = 1
                H = test_set.graph.entity_size + 1
            ranks += [[h, r, t, L, H]]
        ranks = torch.tensor(ranks, dtype=torch.long, device=self.device)
            
        if self.world_size > 1:
            ranks = comm.cat(ranks)
        
        query2LH = dict()
        for h, r, t, L, H in ranks.data.cpu().numpy().tolist():
            query2LH[(h, r, t)] = (L, H)
            
        hit1, hit3, hit10, mr, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        for (L, H) in query2LH.values():
            if expectation:
                for rank in range(L, H):
                    if rank <= 1:
                        hit1 += 1.0 / (H - L)
                    if rank <= 3:
                        hit3 += 1.0 / (H - L)
                    if rank <= 10:
                        hit10 += 1.0 / (H - L)
                    mr += rank / (H - L)
                    mrr += 1.0 / rank / (H - L)
            else:
                rank = H - 1
                if rank <= 1:
                    hit1 += 1
                if rank <= 3:
                    hit3 += 1
                if rank <= 10:
                    hit10 += 1
                mr += rank
                mrr += 1.0 / rank
            
        hit1 /= len(ranks)
        hit3 /= len(ranks)
        hit10 /= len(ranks)
        mr /= len(ranks)
        mrr /= len(ranks)

        if comm.get_rank() == 0:
            logging.info('Data : {}'.format(len(query2LH)))
            logging.info('Hit1 : {:.6f}'.format(hit1))
            logging.info('Hit3 : {:.6f}'.format(hit3))
            logging.info('Hit10: {:.6f}'.format(hit10))
            logging.info('MR   : {:.6f}'.format(mr))
            logging.info('MRR  : {:.6f}'.format(mrr))

        return mrr

    def load(self, checkpoint, load_optimizer=True):
        """
        Load a checkpoint from file.
        Parameters:
            checkpoint (file-like): checkpoint file
            load_optimizer (bool, optional): load optimizer state or not
        """
        if comm.get_rank() == 0:
            logging.info("Load checkpoint from %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(state["model"])

        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        comm.synchronize()

    def save(self, checkpoint):
        """
        Save checkpoint to file.
        Parameters:
            checkpoint (file-like): checkpoint file
        """
        if comm.get_rank() == 0:
            logging.info("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        if self.rank == 0:
            state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            }
            torch.save(state, checkpoint)

        comm.synchronize()

class TrainerGenerator(object):

    def __init__(self, model, model_design, loss_mode, is_scheduled_sampling, gpu): # model: 模型 object, model_design: ori or multitask sharing / mmoe
        self.model = model
        self.model_design = model_design
        self.loss_mode = loss_mode
        self.is_scheduled_sampling = is_scheduled_sampling

        self.task_alpha = {
            "main":            {"use_alpha": False},
            "aux_rel_cluster": {"use_alpha": False},
            "aux_ent_type":    {"use_alpha": True if is_scheduled_sampling else False},
        }

        if gpu is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(gpu)

        model = model.cuda(self.device)

    def compute_total_loss(
        self,
        main_loss: torch.Tensor,
        aux_losses: list[torch.Tensor],
        epoch: int,
        max_epoch: int = 20,
        warmup_epochs: int = 10,
    ):
        n_aux = len(aux_losses)
        if n_aux == 0:
            return main_loss

        if self.loss_mode == 'fixed':
            weights = [0.1, 0.1]              # 自己在此列出輔助權重
            assert len(weights) == len(aux_losses)

            aux_part = sum(w * l for w, l in zip(weights, aux_losses))
            return 0.8 * main_loss + aux_part

        elif self.loss_mode == 'warmup':
            if epoch < warmup_epochs:
                return main_loss
            aux_w_each = 0.05 / n_aux      # 例：總共 0.05
            main_w = 0.95
            aux_part = sum(aux_w_each * l for l in aux_losses)
            return main_w * main_loss + aux_part

        elif self.loss_mode == 'schedule':
            alpha = min(epoch / max_epoch, 1.0)   # alpha 越接近 1 → main 比重越大
            aux_w_each = (1 - alpha) / n_aux
            aux_part = sum(aux_w_each * l for l in aux_losses)
            return alpha * main_loss + aux_part

        elif self.loss_mode == 'adaptive':
            all_losses = torch.stack([main_loss] + aux_losses)
            weights = all_losses / (all_losses.sum() + 1e-8)  # 每項占比
            return (weights[0] * main_loss +
                    sum(w * l for w, l in zip(weights[1:], aux_losses)))

        else:
            raise ValueError(f"Unknown loss_mode: {self.loss_mode}")


    def compute_rule_accuracies_from_logits(self, logits, targets, padding_idx):
        probs = F.softmax(logits, dim=-1)
        batch_size, seq_len, num_rel = probs.shape

        probs_flat = probs.view(-1, num_rel)
        targets_flat = targets.view(-1)

        mask = (targets_flat != padding_idx)

        # gt_probs = probs_flat[torch.arange(probs_flat.size(0)), targets_flat]
        # 先建立 mask，排除掉 padding idx
        mask = (targets_flat != padding_idx)

        # 把非法 index 先設為 0（或任何合法 index）
        safe_targets = targets_flat.clone()
        safe_targets[~mask] = 0  # 或 any valid index like 0

        # 再 gather ground truth prob
        gt_probs = probs_flat[torch.arange(probs_flat.size(0)), safe_targets]

        # 再把 mask 應用上去
        gt_probs = gt_probs * mask.float()

        # gt_probs = gt_probs * mask

        gt_probs = gt_probs.view(batch_size, seq_len)
        mask = mask.view(batch_size, seq_len)

        rule_scores = gt_probs.sum(dim=1)
        valid_lens = mask.sum(dim=1)

        rule_accs = (rule_scores / (valid_lens + 1e-8)).tolist()
        return rule_accs
    
    def schedule_alpha(self, epoch: int, t1: int = 10, t2: int = 50) -> float:
        if epoch <= t1:
            return 0.0
        elif epoch <= t2:
            return (epoch - t1) / float(t2 - t1)
        else:
            return 1.0

    def train(self, rule_set,
            num_epoch=10000, lr=1e-3,
            print_every=100, batch_size=512):

        if comm.get_rank() == 0:
            logging.info('>>>>> Generator: Training')

        model = self.model
        model.train()

        dataloader = torch_data.DataLoader(
            rule_set, batch_size, shuffle=True,
            collate_fn=rule_set.collate_fn)
        iterator  = Iterator(dataloader)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # -------- 多任務設定 --------
        running = {k: 0.0 for k in ["total", "main", "aux_rel_cluster", "aux_ent_type"]}

        for epoch in range(1, num_epoch + 1):
            alpha = self.schedule_alpha(epoch)  

            batch = next(iterator)
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.cuda(self.device, non_blocking=True)

            hidden = self.zero_state(batch["sequence"].size(0))

            if self.model_design == "ori":
                main_loss  = model.loss(batch, hidden)       # 單任務
                aux_losses = []                              # 無輔助
            else:
                # ---------- 主任務 ----------
                main_loss = model.loss(batch, hidden, task="main")

                aux_losses = []
                for task in ("aux_rel_cluster", "aux_ent_type"):
                    cfg = self.task_alpha[task]
                    kwargs = {"alpha": alpha} if cfg["use_alpha"] else {}
                    aux_losses.append(model.loss(batch, hidden, task=task, **kwargs))

            # ----- 組合總 loss -----
            total_step_loss = self.compute_total_loss(main_loss, aux_losses, epoch=epoch)

            # ----- 反向傳播 -----
            total_step_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # ----- 累計 & log -----
            running["total"]            += total_step_loss.item()
            running["main"]             += main_loss.item()
            if aux_losses:
                running["aux_rel_cluster"] += aux_losses[0].item()
                running["aux_ent_type"]    += aux_losses[1].item()

            if epoch % print_every == 0 and comm.get_rank() == 0:
                avg = {k: v / print_every for k, v in running.items()}
                logging.info(
                    f"[E{epoch:05d}] "
                    f"Total {avg['total']:.6f} | "
                    f"Main {avg['main']:.6f} | "
                    f"RelClus {avg['aux_rel_cluster']:.6f} | "
                    f"EntType {avg['aux_ent_type']:.6f}"
                )
                running = {k: 0.0 for k in running}  # reset
    
    def zero_state(self, batch_size): 
        state_shape = (self.model.num_layers, batch_size, self.model.hidden_dim)
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False, device=self.device)
        return (h0, c0)
    
    @torch.no_grad()
    def log_probability(self, rules):
        if rules == []:
            return []
        
        model = self.model
        model.eval()

        rules = [rule + [model.ending_idx] for rule in rules]
        max_len = max([len(rule) for rule in rules])
        for k in range(len(rules)):
            rule_len = len(rules[k])
            for i in range(max_len - rule_len):
                rules[k] += [model.padding_idx]
        rules = torch.tensor(rules, dtype=torch.long, device=self.device)
        inputs = rules[:, :-1]
        target = rules[:, 1:]
        n, l = target.size(0), target.size(1)
        mask = (target != model.padding_idx)
        hidden = self.zero_state(inputs.size(0))
        logits, hidden = model(inputs, inputs[:, 0], hidden)
        logits = torch.log_softmax(logits, -1)
        logits = logits * mask.unsqueeze(-1)
        target = (target * mask).unsqueeze(-1)
        log_prob = torch.gather(logits, -1, target).squeeze(-1) * mask
        log_prob = log_prob.sum(-1)
        return log_prob.data.cpu().numpy().tolist()

    @torch.no_grad()
    def next_relation_log_probability(self, rule, temperature):
        """
        rule: List of relation IDs so far (e.g., [12, 4, 7])
        returns: log probability of next relation (shape: [label_size])
        """
        model = self.model
        device = self.device

        inputs = torch.tensor(rule, dtype=torch.long, device=device).unsqueeze(0)  # shape [1, L]
        relation = torch.tensor([rule[0]], dtype=torch.long, device=device)       # head relation
        mask = torch.ones_like(inputs, dtype=torch.bool)
        weight = torch.ones_like(inputs, dtype=torch.float)

        seq  = torch.tensor([rule], dtype=torch.long, device=self.device)   # [1 , L]
        head = seq[:, 0]                                                    # [1]

        B, L = seq.shape
        T    = self.model.type_size        # 你在模型 __init__ 時存好的 type vocab 大小

        batch = {
            "sequence": inputs,     # [1, L]
            "relation": relation,   # [1]
            "mask": mask,           # [1, L]
            "weight": weight,       # [1, L]
            "aux_ent_type_multihot" : torch.zeros(B, L, T, device=seq.device),
        }

        if self.model_design == "ori":
            logits, _ = model(batch, hidden=None)
        else: # multitask sharing / mmoe
            logits, _, _ = model(batch, hidden=None, task="main")  # logits: [1, L, label_size]
        logits = logits[:, -1, :]   # 取最後一個 token 的 output → shape: [1, label_size]
        logits = logits.squeeze(0) / temperature

        log_prob = torch.log_softmax(logits, dim=-1)  # shape: [label_size]
        return log_prob
    
    @torch.no_grad()
    def beam_search(self, num_samples, max_len, temperature=0.2):
        if comm.get_rank() == 0:
            logging.info('>>>>> Generator: Rule generation with beam search')
        model = self.model
        model.eval()
        
        max_len += 1
        all_rules = []
        for relation in range(model.num_relations):
            found_rules = []
            prev_rules = [[[relation], 0]]
            for k in range(max_len):
                current_rules = list()
                for _i, (rule, score) in enumerate(prev_rules):
                    assert rule[-1] != model.ending_idx
                    log_prob = self.next_relation_log_probability(rule, temperature)
                    for i in (range(model.label_size) if (k + 1) != max_len else [model.ending_idx]):
                        new_rule = rule + [i]
                        new_score = score + log_prob[i]
                        (current_rules if i != model.ending_idx else found_rules).append((new_rule, new_score))
                    
                prev_rules = sorted(current_rules, key=lambda x:x[1], reverse=True)[:num_samples]
                found_rules = sorted(found_rules, key=lambda x:x[1], reverse=True)[:num_samples]

            ret = [rule[0:-1] + [score] for rule, score in found_rules]
            all_rules += ret
        return all_rules
    
    @torch.no_grad()
    def sample(self, num_samples, max_len, temperature=1.0):
        if comm.get_rank() == 0:
            logging.info('>>>>> Generator: Rule generation with sampling')
        model = self.model
        model.eval()

        all_rules = []
        for relation in range(model.num_relations):
            rules = torch.zeros([num_samples, max_len + 1], dtype=torch.long, device=self.device) + model.ending_idx
            log_probabilities = torch.zeros([num_samples, max_len + 1], device=self.device)
            head = torch.tensor([relation for k in range(num_samples)], dtype=torch.long, device=self.device)

            rules[:, 0] = relation
            hidden = self.zero_state(num_samples)

            for pst in range(max_len):
                inputs = rules[:, pst].unsqueeze(-1)

                batch = {
                    "sequence": inputs,                  # [B, 1]
                    "relation": head,                    # [B]
                    "mask": torch.ones_like(inputs, dtype=torch.bool),     # [B, 1]
                    "weight": torch.ones_like(inputs, dtype=torch.float),  # [B, 1]
                }
                B, L = inputs.shape
                T     = model.type_size               # 你在 MultitaskMMOE.__init__ 裡存的
                aux_mh = torch.zeros(B, L, T, device=inputs.device)   # [B, L, T]
                batch["aux_ent_type_multihot"] = aux_mh
                
                if self.model_design == "ori":
                    logits, hidden = model(batch, hidden)
                else: # multitask
                    logits, loss, hidden = model(batch, hidden, task="main")

                logits /= temperature
                log_probability = torch.log_softmax(logits.squeeze(1), dim=-1)
                probability = torch.softmax(logits.squeeze(1), dim=-1)
                sample = torch.multinomial(probability, 1)
                log_probability = log_probability.gather(1, sample)

                mask = (rules[:, pst] != model.ending_idx)
                
                rules[mask, pst + 1] = sample.squeeze(-1)[mask]
                log_probabilities[mask, pst + 1] = log_probability.squeeze(-1)[mask]

            length = (rules != model.ending_idx).sum(-1).unsqueeze(-1) - 1
            formatted_rules = torch.cat([length, rules], dim=1)

            log_probabilities = log_probabilities.sum(-1)

            formatted_rules = formatted_rules.data.cpu().numpy().tolist()
            log_probabilities = log_probabilities.data.cpu().numpy().tolist()
            for k in range(num_samples):
                length = formatted_rules[k][0]
                formatted_rules[k] = formatted_rules[k][1: 2 + length] + [log_probabilities[k]]

            rule_set = set([tuple(rule) for rule in formatted_rules])
            formatted_rules = [list(rule) for rule in rule_set]

            all_rules += formatted_rules

        return all_rules
    
    def load(self, checkpoint):
        """
        Load a checkpoint from file.
        Parameters:
            checkpoint (file-like): checkpoint file
            load_optimizer (bool, optional): load optimizer state or not
        """
        if comm.get_rank() == 0:
            logging.info("Load checkpoint from %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state["model"])

    def save(self, checkpoint):
        """
        Save checkpoint to file.
        Parameters:
            checkpoint (file-like): checkpoint file
        """
        if comm.get_rank() == 0:
            logging.info("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = {
            "model": self.model.state_dict()
        }
        torch.save(state, checkpoint)    
