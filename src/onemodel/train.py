"""Training entry point: python -m onemodel.train CONFIG [--dry-run]."""

from dataclasses import replace

from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from onemodel.cli import make_parser, print_config
from onemodel.config import load_config
from onemodel.trainers.utils import create_run_dir, save_run_metadata


def main() -> None:
    parser = make_parser("Train a unified recommendation model.")
    args = parser.parse_args()
    experiment = load_config(args.config, args.gin)
    if args.dry_run:
        print_config(experiment)
        return
    set_seed(experiment.training.seed)
    try:
        model = experiment.model()
        train_dataset = experiment.dataset(
            root=experiment.data.root, split=experiment.data.train_split
        )
        valid_dataset = experiment.dataset(
            root=experiment.data.root, split=experiment.data.valid_split
        )
        collator = experiment.collator()
        loader_kwargs = {
            "batch_size": experiment.data.batch_size,
            "num_workers": experiment.data.num_workers,
            "collate_fn": collator,
        }
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
        trainer = experiment.trainer(
            model=model, config=experiment.training, objective=experiment.objective
        )
        run_dir = create_run_dir(experiment, trainer.accelerator)
        if trainer.accelerator.is_main_process:
            save_run_metadata(run_dir, experiment, args.config, args.gin)
        trainer.accelerator.wait_for_everyone()
        trainer.fit(
            train_loader,
            run_dir,
            valid_loader=valid_loader,
            evaluator=experiment.evaluator,
            evaluation_config=replace(
                experiment.evaluation, split=experiment.data.valid_split, checkpoint=None
            ),
        )
    except NotImplementedError as error:
        parser.exit(2, f"Project skeleton: {error}\n")


if __name__ == "__main__":
    main()
