import argparse
import importlib
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class RandomTokenDataset(Dataset):
    def __init__(self, samples, seq_len, vocab_size):
        self.samples = samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.samples

    def __getitem__(self, _):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        return {"input_ids": tokens, "labels": tokens.clone()}


def load_model(model_name, **kwargs):
    module_path = f"llm_models.{model_name}"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        if exc.name == module_path:
            available = sorted(p.stem for p in Path("llm_models").glob("*.py") if p.name != "__init__.py")
            raise SystemExit(f"Unknown model '{model_name}'. Available models: {available}") from exc
        raise

    if not hasattr(module, "build_model"):
        raise SystemExit(f"{module_path} must define build_model(**kwargs)")
    return module.build_model(**kwargs)


def parse_args():
    import deepspeed

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="tiny_gpt", help="Loads llm_models/<model_name>.py")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--dataset_samples", type=int, default=10000)
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", "-1")))
    parser = deepspeed.add_config_arguments(parser)
    parser.set_defaults(deepspeed_config="ds_config_zero3.json")
    return parser.parse_args()


def main():
    import deepspeed

    args = parse_args()
    deepspeed.init_distributed()

    rank = torch.distributed.get_rank()
    torch.cuda.set_device(args.local_rank)

    model = load_model(args.model_name, vocab_size=args.vocab_size, seq_len=args.seq_len)
    dataset = RandomTokenDataset(args.dataset_samples, args.seq_len, args.vocab_size)
    loader = DataLoader(dataset, batch_size=None, shuffle=False)

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=loader,
        config=args.deepspeed_config,
    )

    engine.train()
    data_iter = iter(engine.training_dataloader)

    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(engine.training_dataloader)
            batch = next(data_iter)

        batch = {key: value.to(engine.device) for key, value in batch.items()}
        loss = engine(**batch)["loss"]
        engine.backward(loss)
        engine.step()

        if rank == 0 and step % 10 == 0:
            print(f"step={step} loss={loss.item():.4f}")

        if step % args.save_every == 0:
            engine.save_checkpoint(args.output_dir, tag=f"step-{step}")

    if rank == 0:
        print("done")


if __name__ == "__main__":
    main()
