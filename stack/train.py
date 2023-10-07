import torch
import numpy as np
from brainbox import trainer

from stack import models, loss


class Trainer(trainer.Trainer):

    def __init__(self, root, model, train_dataset, n_epochs, batch_size, lr, loss_type, lam, lam0=None, crop=10, detach_target=False, device="cuda", id=None, pred_steps=1, lam_activity=10**-7):
        super().__init__(root, model, train_dataset, n_epochs, batch_size, lr, optimizer_func=torch.optim.Adam, scheduler_func=None, scheduler_kwargs={}, loader_kwargs={"shuffle": True}, device=device, grad_clip_type=None, grad_clip_value=0, id=id)
        self._loss_type = loss_type
        self._lam = lam
        self._lam0 = lam0
        self._crop = crop
        self._detach_target = detach_target
        self._pred_steps = pred_steps
        self._lam_activity = lam_activity

        if loss_type == "prediction":
            self._loss = loss.PredictionLoss(model.get_params, lam, self._lam0, crop, detach_target, pred_steps=pred_steps)
        elif loss_type == "compression":
            self._loss = loss.SparseCompressionLoss(model.get_params, lam, self._lam0, crop, detach_target, lam_activity=lam_activity)
        elif loss_type == "slowness":
            self._loss = loss.SlownessLoss(model.get_params, lam, self._lam0, crop, detach_target)

        self._min_loss = np.inf

    @staticmethod
    def load_model(root, model_id, override_kwargs={}):

        def model_loader(hyperparams):
            model_params = hyperparams["model"]
            del model_params["name"]
            del model_params["weight_initializers"]

            for key, value in override_kwargs.items():
                print(f"{key}-{value}")
                model_params[key] = value

            return models.StackModel(**model_params)

        return trainer.load_model(root, model_id, model_loader)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "loss_type": self._loss_type, "lam": self._lam, "lam0": self._lam0, "crop": self._crop, "detach_target": self._detach_target, "pred_steps": self._pred_steps, "lam_activity": self._lam_activity}
    
    def on_epoch_complete(self, save):
        _train_loss = self.log["train_loss"][-1]
        n_epochs = len(self.log["train_loss"])
        if n_epochs % 10 == 0:
            print(f"Completed {n_epochs}...")

        # Check if new minimum loss has been reached
        if _train_loss < self._min_loss:
            print(f"Saving model train_loss={_train_loss:.4f} < min_loss={self._min_loss:.4f}")
            self._min_loss = _train_loss

            if save:
                self.save_model()

        # Save logs and hyperparams
        if save:
            self.save_model_log()
            self.save_hyperparams()

    def loss(self, output, target, model):
        return self._loss(output)

    def train(self, save=True):
        super().train(save)
