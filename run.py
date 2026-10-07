import argparse
import os

os.environ["MKL_THREADING_LAYER"] = "GNU"
import random
import numpy as np
import torch
from exp.exp_main import Exp_Main

# Modes: train_test or test_only.
RUN_MODE = "train_test"

VALID_RUN_MODES = {"train_test", "test_only"}
CHECKPOINT_FILENAME = "checkpoint.pth"

# Disable TF32 for FP32 experiments.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    if RUN_MODE not in VALID_RUN_MODES:
        raise ValueError(
            f'RUN_MODE={RUN_MODE!r} 无效，可选值为：{sorted(VALID_RUN_MODES)}'
        )

    # fix_seed(2022)
    # fix_seed(2023)
    fix_seed(2024)
    # fix_seed(2025)
    # fix_seed(2026)

    parser = argparse.ArgumentParser(
        description='AnStaEF for Long Term Time Series Forecasting'
    )

    # Experiment
    parser.add_argument(
        '--model_id', type=str, required=True, default='test', help='model id'
    )
    parser.add_argument(
        '--model', type=str, required=True, default='AnStaEF', help='model name'
    )

    # Data
    parser.add_argument(
        '--data', type=str, required=True, default='ETTh1', help='dataset type'
    )
    parser.add_argument(
        '--root_path', type=str, default='./dataset/', help='root path of the data file'
    )
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument(
        '--features', type=str, default='M', help='forecasting task, options:[M, S, MS]'
    )
    parser.add_argument(
        '--target', type=str, default='OT', help='target feature in S or MS task'
    )
    parser.add_argument(
        '--checkpoints',
        type=str,
        default='./checkpoints/',
        help='location of model checkpoints',
    )

    parser.add_argument(
        '--embed',
        type=str,
        default='timeF',
        help='time features encoding, options:[timeF, fixed, learned]',
    )
    parser.add_argument(
        '--freq',
        type=str,
        default='h',
        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]',
    )

    # Forecasting
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument(
        '--pred_len', type=int, default=96, help='prediction sequence length'
    )

    # Model
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    # Expert fusion and routing
    parser.add_argument('--num_experts', type=int, default=8, help='number of experts')
    parser.add_argument('--top_k', type=int, default=2, help='top k experts to route')
    parser.add_argument(
        '--router_temperature',
        type=float,
        default=1.0,
        help='router softmax temperature',
    )
    parser.add_argument(
        '--router_jitter',
        type=float,
        default=0.01,
        help='input jitter for router stability',
    )
    parser.add_argument(
        '--router_smoothing',
        type=float,
        default=0.02,
        help='training-time uniform smoothing coefficient for selected Top-k weights',
    )
    parser.add_argument(
        '--temporal_mixer_type',
        type=str,
        default='segment_attn',
        help='temporal processing module: segment_attn (SLTI) or conv',
    )
    parser.add_argument(
        '--segment_size',
        type=int,
        default=4,
        help='number of consecutive patches per SLTI segment',
    )
    parser.add_argument(
        '--segment_attn_heads',
        type=int,
        default=4,
        help='number of attention heads in SLTI',
    )

    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')

    # Loss
    parser.add_argument(
        '--lambda_mae', type=float, default=0.5, help='weight for MAE loss'
    )
    parser.add_argument(
        '--lambda_balance',
        type=float,
        default=0.001,
        help='weight for load balancing loss',
    )
    parser.add_argument(
        '--lambda_ortho', type=float, default=0.01, help='weight for orthogonality loss'
    )
    parser.add_argument(
        '--lambda_z', type=float, default=0.001, help='weight for router z loss'
    )
    parser.add_argument(
        '--lambda_entropy',
        type=float,
        default=0.001,
        help='weight for router entropy regularization',
    )
    parser.add_argument(
        '--aux_warmup_epochs',
        type=int,
        default=5,
        help='warmup epochs for auxiliary losses',
    )
    parser.add_argument(
        '--router_temp_start',
        type=float,
        default=1.5,
        help='start temperature for router annealing',
    )
    parser.add_argument(
        '--router_temp_end',
        type=float,
        default=0.8,
        help='end temperature for router annealing',
    )

    # Training
    parser.add_argument(
        '--num_workers', type=int, default=0, help='data loader num workers'
    )
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=40, help='train epochs')
    parser.add_argument(
        '--batch_size', type=int, default=32, help='batch size of train input data'
    )
    parser.add_argument(
        '--patience', type=int, default=5, help='early stopping patience'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=0.0001, help='optimizer learning rate'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.0001, help='optimizer weight decay'
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0,
        help='max gradient norm, <=0 to disable',
    )
    parser.add_argument(
        '--use_amp',
        action='store_true',
        help='use automatic mixed precision training',
        default=False,
    )
    parser.add_argument(
        '--use_ema', type=int, default=1, help='use ema model averaging'
    )
    parser.add_argument('--ema_decay', type=float, default=0.999, help='ema decay rate')

    # Device
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument(
        '--use_multi_gpu', action='store_true', help='use multiple gpus', default=False
    )
    parser.add_argument(
        '--devices', type=str, default='0,1', help='device ids of multile gpus'
    )

    parser.add_argument(
        '--use_anchor', type=int, default=1, help='use anchor stream or not'
    )
    parser.add_argument(
        '--use_cvce',
        type=int,
        default=1,
        help='enable Cross-Variable Context Exchange (CVCE)',
    )
    parser.add_argument(
        '--cvce_bottleneck',
        type=int,
        default=2,
        help='CVCE query-bank size multiplier; total queries = number of variables * cvce_bottleneck',
    )
    parser.add_argument(
        '--cvce_heads',
        type=int,
        default=4,
        help='number of attention heads in CVCE MHA',
    )
    args = parser.parse_args()

    if not os.path.exists(args.root_path):
        candidate = None
        if args.root_path.startswith('.data'):
            candidate = './data'
        elif args.root_path in ['data', 'data/']:
            candidate = './data'
        if candidate and os.path.exists(candidate):
            args.root_path = candidate

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    for ii in range(args.itr):
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_dm{}_exp{}_top{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.num_experts,
            args.top_k,
            ii,
        )

        exp = Exp_Main(args)

        if RUN_MODE == "train_test":
            print(
                '>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting)
            )
            exp.train(setting)

            print(
                '>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting)
            )
            # train() has already restored the best checkpoint.
            exp.test(setting, test=0)

        elif RUN_MODE == "test_only":
            checkpoint_path = os.path.join(
                args.checkpoints, setting, CHECKPOINT_FILENAME
            )

            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    '没有找到待测试的 checkpoint：\n'
                    f'{checkpoint_path}\n'
                    '请确认 RUN_MODE、model_id、数据集、预测长度、d_model、'
                    'num_experts、top_k 以及 itr 与训练时完全一致。'
                )

            print(
                '>>>>>>>loading checkpoint : {}<<<<<<<<<<<<<<<<<<<<<<'.format(
                    checkpoint_path
                )
            )

            state_dict = torch.load(checkpoint_path, map_location=exp.device)
            exp.model.load_state_dict(state_dict, strict=True)

            print(
                '>>>>>>>testing only : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting)
            )
            exp.test(setting, test=0)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
