"""Private synchronous vLLM worker. Started by ColocatedVLLMRollout."""

from multiprocessing.connection import Connection
import os
import sys
import traceback

# vLLM's cumem_allocator abi3 wheel returns Py_None from python_create_and_map /
# python_unmap_and_release WITHOUT incrementing its refcount: the inlined
# Py_INCREF was compiled against Python 3.12+ headers, where None is immortal.
# Every sleep/wake cycle therefore shaves None's refcount by one per call until
# CPython aborts with "Fatal Python error: none_dealloc: deallocating None"
# (~1300 cycles here, scaling with the number of sleep-mode handles). Python
# 3.12 is immune by construction; under 3.11 keep the count far above the drain.
_NONE_TOPPUP_SLACK = 100_000
_toppup_broken = False


def _none_refcount():
    return sys.getrefcount(None)


def _top_up_none_refcount(floor):
    """Re-add the None references drained by the cumem leak; 0 if not needed."""
    global _toppup_broken
    if _toppup_broken:
        return 0
    missing = floor - _none_refcount()
    if missing <= 0:
        return 0
    try:
        import ctypes

        incref = ctypes.pythonapi.Py_IncRef
        incref.argtypes = [ctypes.py_object]
        incref.restype = None
        for _ in range(missing):
            incref(None)
    except Exception:
        _toppup_broken = True
        print(
            f"[GRPO vLLM worker] none-refcount top-up disabled: {traceback.format_exc()}",
            flush=True,
        )
        return 0
    return missing


class RolloutLogitsProcessor:
    def __init__(self, banned):
        self.banned = banned

    def __call__(self, token_ids, logits):
        if __package__:
            from .vllm_rollout import finite_rollout_logits
        else:
            from vllm_rollout import finite_rollout_logits

        return finite_rollout_logits(logits, self.banned)


def main(fd):
    connection = Connection(fd)
    llm = None
    none_floor = _none_refcount() + _NONE_TOPPUP_SLACK
    commands = 0
    topped_up = 0
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
            commands += 1
            added = _top_up_none_refcount(none_floor)
            if added:
                topped_up += added
            if added > _NONE_TOPPUP_SLACK // 10 or commands % 200 == 0:
                print(
                    f"[GRPO vLLM worker] command={commands} "
                    f"none_refcount={_none_refcount()} topped_up_total={topped_up}",
                    flush=True,
                )
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
