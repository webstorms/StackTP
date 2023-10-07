import os

root = "/home/luketaylor/PycharmProjects/StackTP"  # Specify path to project
dataset_path = "/home/datasets/natural"  # Specify path to dataset

os.system(f"python {root}/scripts/train.py --root={root} --dataset_path={dataset_path} --filtered=False --loss_type=slowness --lam=1e-6 --lam0=1e-4 --detach_target=True --id=slowness_4-6_True_False --lr=0.0001 --n_epochs=300")
os.system(f"python {root}/scripts/train.py --root={root} --dataset_path={dataset_path} --filtered=False --loss_type=compression --lam=1e-6 --lam0=1e-4 --detach_target=True --id=compression_4-6_True_False_8 --lr=0.0001 --n_epochs=300 --lam_activity=1e-8")
os.system(f"python {root}/scripts/train.py --root={root} --dataset_path={dataset_path} --filtered=False --loss_type=prediction --lam=1e-6 --lam0=1e-4 --detach_target=True --id=prediction_4-6_True_False_2 --lr=0.0001 --n_epochs=300 --pred_steps=2")
