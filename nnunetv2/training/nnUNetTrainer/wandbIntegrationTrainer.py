import torch
import wandb

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
class nnUNetTrainer_Wandb_Logger(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        wandb.init(
                project="pinkcc",
                name=f"Dataset{self.dataset_name}_{self.configuration}_{self.fold}",
                config={
                    "plans": self.plans,
                    "configuration": self.configuration,
                    "fold": self.fold,
                    "batch_size": self.batch_size,
                    "patch_size": self.patch_size,
                    "num_epochs": self.num_epochs
                }
            )

    def on_epoch_end(self):
        super().on_epoch_end()

        wandb.log({"epoch": self.current_epoch,
                   "train_loss": self.all_tr_losses[-1],
                   "val_loss": self.all_val_losses[-1] if len(self.all_val_losses) > 0 else None,
                   "learning_rate": self.optimizer.param_groups[0]['lr']})
       
    