import sys
import ast
import logging
import argparse

from stack import dataset, models, train

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def eval(v):
    return ast.literal_eval(v)


def get_model(args):
    return models.StackModel(recurrent=False)


def get_dataset(args):
    return dataset.NaturalDataset(root=args.dataset_path, train=True, dt=args.dt, flip=True, filtered=eval(args.filtered))


def get_trainer(args, model, train_dataset):
    return train.Trainer(f"{args.root}/results", model, train_dataset, args.n_epochs, args.batch_size, args.lr, args.loss_type, args.lam, args.lam0, crop=10, detach_target=eval(args.detach_target), id=args.id, pred_steps=args.pred_steps, lam_activity=args.lam_activity)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=".")

    # Dataset
    parser.add_argument('--dataset_path', type=str, default=".")
    parser.add_argument('--dt', type=int, default=30)
    parser.add_argument('--flip', type=str, default="True")
    parser.add_argument('--filtered', type=str, default="True")

    # Trainer
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--loss_type', type=str, default="prediction")
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--lam0', type=float, default=None)
    parser.add_argument('--detach_target', type=str, default="False")
    parser.add_argument('--id', type=str, default="")
    parser.add_argument('--pred_steps', type=int, default=1)
    parser.add_argument('--lam_activity', type=float, default=10**-7)

    args = parser.parse_args()

    train_dataset = get_dataset(args)
    model = get_model(args)
    model_trainer = get_trainer(args, model, train_dataset)
    model_trainer.train()


if __name__ == '__main__':
    main()
