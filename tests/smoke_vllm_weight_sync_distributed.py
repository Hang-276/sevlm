"""Opt-in two-rank checks for ZeRO-3 and FSDP2 policy-weight gathering.

torchrun --standalone --nproc_per_node=2 tests/smoke_vllm_weight_sync_distributed.py
"""

import os

import torch
import torch.distributed as dist

from open_r1.trainer.vllm_rollout import iter_policy_weights


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    try:
        import deepspeed
        from torch.distributed._composable.fsdp import fully_shard

        def make_model():
            torch.manual_seed(37)
            return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 4)).to(device)

        reference = {n: p.detach().cpu().clone() for n, p in make_model().named_parameters()}
        config = {"train_batch_size": 2, "zero_optimization": {"stage": 3}, "bf16": {"enabled": False}}
        with deepspeed.zero.Init(config_dict_or_path=config, dtype=torch.float32):
            zero_model = make_model()
        # ZeRO initializes on CUDA rather than CPU, so seeded random initializers
        # differ. Load the same reference weights through the normal DS context.
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
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
