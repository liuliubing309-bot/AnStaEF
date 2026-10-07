import os
import copy
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider
from models import AnStaEF
from utils.losses import AnStaEFLoss
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, Visualizer

warnings.filterwarnings('ignore')


class Exp_Main(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.criterion = self._select_criterion()
        self.optimizer = self._select_optimizer()
        self.ema_model = self._build_ema_model()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model = AnStaEF.Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _model_module(self):
        return (
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        )

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        return model_optim

    def _select_criterion(self):
        criterion = AnStaEFLoss(
            lambda_mae=self.args.lambda_mae,
            lambda_balance=self.args.lambda_balance,
            lambda_ortho=self.args.lambda_ortho,
            lambda_z=self.args.lambda_z,
            lambda_entropy=self.args.lambda_entropy,
        )
        return criterion

    def _build_ema_model(self):
        if not self.args.use_ema:
            return None
        ema_model = copy.deepcopy(self.model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        return ema_model

    def _update_ema(self):
        if not self.args.use_ema or self.ema_model is None:
            return
        decay = self.args.ema_decay
        with torch.no_grad():
            msd = self.model.state_dict()
            esd = self.ema_model.state_dict()
            for k, v in esd.items():
                if v.dtype.is_floating_point:
                    v.mul_(decay).add_(msd[k], alpha=1 - decay)
                else:
                    v.copy_(msd[k])

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        aux_warmup_epochs = max(1, int(getattr(self.args, 'aux_warmup_epochs', 5)))
        router_temp_start = float(
            getattr(self.args, 'router_temp_start', self.args.router_temperature)
        )
        router_temp_end = float(
            getattr(self.args, 'router_temp_end', self.args.router_temperature)
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            moe_diag = {
                'candidate_top1_counts': np.zeros(self.args.num_experts),
                'total_tokens': 0,
                'router_entropy_sum': 0.0,
                'router_entropy_count': 0,
                'l_balance_sum': 0.0,
                'l_z_sum': 0.0,
                'logsumexp_max': -float('inf'),
                'logsumexp_sum': 0.0,
                'logsumexp_count': 0,
                'num_batches': 0,
            }
            aux_scale = min(1.0, float(epoch + 1) / float(aux_warmup_epochs))

            self.model.train()
            epoch_time = time.time()
            model_core = self._model_module()
            if hasattr(model_core, 'router') and hasattr(
                model_core.router, 'temperature'
            ):
                if self.args.train_epochs > 1:
                    progress = float(epoch) / float(self.args.train_epochs - 1)
                else:
                    progress = 1.0
                current_temp = (
                    router_temp_start + (router_temp_end - router_temp_start) * progress
                )
                model_core.router.temperature = max(1e-3, float(current_temp))

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim = self.optimizer
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, router_logits, candidate_pool = self.model(
                            batch_x, batch_x_mark
                        )

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                            self.device
                        )

                        loss, l_task, l_bal, l_orth, l_z, l_ent = self.criterion(
                            outputs,
                            batch_y,
                            router_logits,
                            candidate_pool,
                            aux_scale=aux_scale,
                        )
                        train_loss.append(loss.item())
                else:
                    outputs, router_logits, candidate_pool = self.model(
                        batch_x, batch_x_mark
                    )

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                    loss, l_task, l_bal, l_orth, l_z, l_ent = self.criterion(
                        outputs,
                        batch_y,
                        router_logits,
                        candidate_pool,
                        aux_scale=aux_scale,
                    )
                    train_loss.append(loss.item())

                # Accumulate routing diagnostics.
                if router_logits is not None:
                    with torch.no_grad():
                        rl = router_logits.detach()
                        rl_flat = rl.reshape(-1, rl.size(-1))
                        n_tok = rl_flat.size(0)
                        top1 = rl_flat.argmax(dim=-1)
                        for e_idx in range(self.args.num_experts):
                            moe_diag['candidate_top1_counts'][e_idx] += (
                                (top1 == e_idx).sum().item()
                            )
                        moe_diag['total_tokens'] += n_tok
                        probs_diag = F.softmax(rl_flat, dim=-1)
                        ent_per_token = -torch.sum(
                            probs_diag * torch.log(probs_diag + 1e-9), dim=-1
                        )
                        moe_diag['router_entropy_sum'] += ent_per_token.sum().item()
                        moe_diag['router_entropy_count'] += n_tok
                        lse = torch.logsumexp(rl_flat, dim=-1)
                        moe_diag['logsumexp_sum'] += lse.sum().item()
                        moe_diag['logsumexp_max'] = max(
                            moe_diag['logsumexp_max'], lse.max().item()
                        )
                        moe_diag['logsumexp_count'] += n_tok
                    moe_diag['l_balance_sum'] += l_bal.item()
                    moe_diag['l_z_sum'] += l_z.item()
                    moe_diag['num_batches'] += 1

                if (i + 1) % 100 == 0:
                    print(
                        f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f} | "
                        f"task: {l_task.item():.5f}, bal: {l_bal.item():.5f}, ortho: {l_orth.item():.5f}, z: {l_z.item():.5f}, ent: {l_ent.item():.5f}"
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * len(train_loader) - i
                    )
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    if self.args.max_grad_norm > 0:
                        scaler.unscale_(model_optim)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.max_grad_norm
                        )
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    if self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.max_grad_norm
                        )
                    model_optim.step()
                self._update_ema()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, vali_data)
            test_loss = self.vali(test_loader, test_data)

            router_temp = (
                getattr(model_core.router, 'temperature', 1.0)
                if hasattr(model_core, 'router')
                else 1.0
            )
            print(
                f"Epoch: {epoch + 1}, Steps: {len(train_loader)} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f} | aux_scale: {aux_scale:.3f} | router_temp: {router_temp:.3f}"
            )

            # Report routing diagnostics.
            if moe_diag['total_tokens'] > 0:
                e_fracs = moe_diag['candidate_top1_counts'] / moe_diag['total_tokens']
                e_frac_var = np.var(e_fracs)
                e_frac_ent = -np.sum(e_fracs * np.log(e_fracs + 1e-9))
                ideal_ent = np.log(self.args.num_experts)
                avg_r_ent = moe_diag['router_entropy_sum'] / max(
                    1, moe_diag['router_entropy_count']
                )
                max_r_ent = np.log(self.args.num_experts)
                load_imb = e_fracs.max() / (e_fracs.min() + 1e-9)
                avg_l_bal = moe_diag['l_balance_sum'] / max(1, moe_diag['num_batches'])
                avg_l_z = moe_diag['l_z_sum'] / max(1, moe_diag['num_batches'])
                avg_lse = moe_diag['logsumexp_sum'] / max(
                    1, moe_diag['logsumexp_count']
                )
                max_lse = moe_diag['logsumexp_max']
                print(
                    f"  [Expert Stream] Candidate Top-1 fractions: {np.array2string(e_fracs, precision=4, separator=', ')}"
                )
                print(
                    f"  [MoE] Frac var: {e_frac_var:.6f} | Frac entropy: {e_frac_ent:.4f}/{ideal_ent:.4f} (ratio: {e_frac_ent/ideal_ent:.3f})"
                )
                print(
                    f"  [MoE] Router avg entropy: {avg_r_ent:.4f}/{max_r_ent:.4f} (ratio: {avg_r_ent/max_r_ent:.3f})"
                )
                print(
                    f"  [MoE] Load imbalance(max/min): {load_imb:.2f} | avg L_balance: {avg_l_bal:.5f}"
                )
                print(
                    f"  [MoE] avg L_z: {avg_l_z:.5f} | logsumexp mean: {avg_lse:.4f} max: {max_lse:.4f}"
                )

            model_to_save = (
                self.ema_model
                if self.args.use_ema and self.ema_model is not None
                else self.model
            )
            early_stopping(vali_loss, model_to_save, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_loader, vali_data):
        model_to_eval = (
            self.ema_model
            if self.args.use_ema and self.ema_model is not None
            else self.model
        )
        model_to_eval.eval()
        total_loss = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # Validation uses task loss only.
                outputs, _, _ = model_to_eval(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = self.criterion.mse_loss(
                    pred, true
                ) + self.args.lambda_mae * self.criterion.l1_loss(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )

        self.model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                outputs, _, _ = self.model(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                preds.append(outputs)
                trues.append(batch_y)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse:{mse}, mae:{mae}')

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
