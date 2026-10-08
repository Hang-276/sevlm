"""Opt-in two-rank checks for ZeRO-3 and FSDP2 policy-weight gathering.

torchrun --standalone --nproc_per_node=2 tests/smoke_vllm_weight_sync_distributed.py
"""

import argparse
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from open_r1.trainer.vllm_rollout import iter_policy_weights
from open_r1.trainer.grpo_trainer import VLMGRPOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("all", "zero3", "fsdp2"), default="all")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    try:
        from torch.distributed._composable.fsdp import fully_shard

        def make_model():
            torch.manual_seed(42)
            return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 4)).to(device)

        reference = {n: p.detach().cpu().clone() for n, p in make_model().named_parameters()}
        if args.backend in {"all", "zero3"}:
            import deepspeed

            config = {"train_batch_size": 2, "zero_optimization": {"stage": 3}, "bf16": {"enabled": False}}
            with deepspeed.zero.Init(config_dict_or_path=config, dtype=torch.float32):
                zero_model = make_model()
            for name, param in zero_model.named_parameters():
                with deepspeed.zero.GatheredParameters([param], modifier_rank=0), torch.no_grad():
                    if dist.get_rank() == 0:
                        param.copy_(reference[name].to(device))
            for name, weight in iter_policy_weights(zero_model):
                torch.testing.assert_close(weight, reference[name])
            with torch.no_grad():
                for param in zero_model.parameters():
                    param.ds_tensor.add_(0.25)
            for name, weight in iter_policy_weights(zero_model):
                torch.testing.assert_close(weight, reference[name] + 0.25)
            print(f"rank={dist.get_rank()} PASS ZeRO-3 initial and updated weight gathering", flush=True)

        if args.backend in {"all", "fsdp2"}:
            model = make_model()
            fully_shard(model)
            for name, weight in iter_policy_weights(model):
                torch.testing.assert_close(weight, reference[name])
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(0.25)
            for name, weight in iter_policy_weights(model):
                torch.testing.assert_close(weight, reference[name] + 0.25)
            print(f"rank={dist.get_rank()} PASS FSDP2 initial and updated weight gathering", flush=True)

        def reduce_flag(flag, reduction):
            assert reduction == "sum"
            dist.all_reduce(flag)
            return flag

        trainer = VLMGRPOTrainer.__new__(VLMGRPOTrainer)
        trainer.accelerator = SimpleNamespace(
            device=device, num_processes=dist.get_world_size(), reduce=reduce_flag,
        )
        trainer._assert_finite_loss_inputs(torch.ones(2, device=device))
        for invalid_rank in range(dist.get_world_size()):
            logps = torch.ones(2, device=device)
            if dist.get_rank() == invalid_rank:
                logps[0] = float("nan")
            try:
                trainer._assert_finite_loss_inputs(logps)
            except FloatingPointError:
                pass
            else:
                raise AssertionError("Every rank must reject the invalid backward graph")
        print(f"rank={dist.get_rank()} PASS synchronized non-finite log-probability rejection", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
