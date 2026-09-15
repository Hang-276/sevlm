"""Private synchronous vLLM worker. Started by ColocatedVLLMRollout."""

from multiprocessing.connection import Connection
import os
import sys
import traceback


class RolloutLogitsProcessor:
    def __init__(self, banned):
        self.banned = banned

    def __call__(self, token_ids, logits):
        import torch

        logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e4, neginf=-1e9)
        if logits.max() <= -1e9:
            logits.zero_()
        banned = [i for i in self.banned if 0 <= i < logits.shape[-1]]
        logits[banned] = float("-inf")
        return logits


def main(fd):
    connection = Connection(fd)
    llm = None
    try:
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            try:
                result = None
                if command == "init":
                    from vllm import LLM

                    llm = LLM(
                        **payload,
                        tensor_parallel_size=1,
                        distributed_executor_backend="uni",
                        enable_sleep_mode=True,
                        enforce_eager=True,
                    )
                    llm.sleep(level=1)
                elif command == "wake":
                    llm.wake_up()
                elif command == "sleep":
                    llm.sleep(level=1)
                elif command == "weight":
                    from safetensors.torch import load

                    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
                    model.load_weights(load(payload).items())
                elif command == "synced":
                    llm.reset_prefix_cache()
                    print(f"[GRPO vLLM worker] policy_step={payload}", flush=True)
                elif command == "generate":
                    from vllm import SamplingParams

                    requests, sampling = payload
                    banned = sampling.pop("banned_token_ids")
                    sampling["logits_processors"] = [RolloutLogitsProcessor(banned)]
                    outputs = llm.generate(requests, SamplingParams(**sampling), use_tqdm=False)
                    result = [
                        dict(
                            prompt_token_ids=o.prompt_token_ids,
                            token_ids=o.outputs[0].token_ids,
                            finish_reason=o.outputs[0].finish_reason,
                        )
                        for o in outputs
                    ]
                else:
                    raise ValueError(f"Unknown worker command: {command}")
                connection.send((True, result))
            except Exception:
                error = traceback.format_exc()
                print(error, flush=True)
                connection.send((False, error))
    except EOFError:
        pass
    finally:
        connection.close()
        if llm is not None:
            from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main(int(sys.argv[1]))
    # Torch 2.6's pluggable allocator can abort while tearing down vLLM sleep
    # pools at interpreter shutdown. This dedicated process owns the CUDA
    # context; let process exit reclaim it after closing IPC/distributed state.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
