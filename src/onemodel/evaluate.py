"""Evaluation entry point using an Accelerator state directory saved by Trainer."""

from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from onemodel.cli import make_parser, print_config
from onemodel.config import load_config


def main() -> None:
    parser = make_parser("Evaluate a unified recommendation model.")
    parser.add_argument("--checkpoint", help="Accelerator state directory produced by training.")
    args = parser.parse_args()
    experiment = load_config(args.config, args.gin)
    if args.checkpoint:
        experiment.evaluation.checkpoint = args.checkpoint
    if args.dry_run:
        print_config(experiment)
        return
    if not experiment.evaluation.checkpoint:
        parser.error("Provide --checkpoint or bind EvaluationConfig.checkpoint in Gin.")
    set_seed(experiment.training.seed)
    try:
        model = experiment.model()
        dataset = experiment.dataset(root=experiment.data.root, split=experiment.evaluation.split)
        dataloader = DataLoader(
            dataset,
            batch_size=experiment.data.batch_size,
            num_workers=experiment.data.num_workers,
            collate_fn=experiment.collator(),
            shuffle=False,
        )
        accelerator = Accelerator(mixed_precision=experiment.training.mixed_precision)
        model, dataloader = accelerator.prepare(model, dataloader)
        accelerator.load_state(experiment.evaluation.checkpoint)
        model.eval()
        metrics = experiment.evaluator(
            model, dataloader, accelerator=accelerator, config=experiment.evaluation
        )
        accelerator.print(metrics)
    except NotImplementedError as error:
        parser.exit(2, f"Project skeleton: {error}\n")


if __name__ == "__main__":
    main()
