import torch
import wandb
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join


from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
class nnUNetTrainer_Wandb_Logger(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)

        wandb.init(
                project="pinkcc",
                # name=f"Dataset{self.dataset_name}_{self.configuration}_{self.fold}",
                config={
                    "plans": self.plans_manager,
                    "configuration": self.configuration_manager,
                    "fold": self.fold,
                    "num_epochs": self.num_epochs
                }
            )
        
    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
                
            wandb.log({
                "epoch": epoch,
                "train_loss": np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4),
                "val_loss": np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4),                
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            dice_scores = self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]
            for class_idx, dice_value in enumerate(dice_scores):
                wandb.log({f'pseudo_dice_class_{class_idx}': np.round(dice_value, decimals=4)}, step=epoch)
            
            if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
                wandb.log({
                    "best_ema": {np.round(self._best_ema, decimals=4)}
                }, step=epoch)
                wandb.log_artifact(join(self.output_folder, 'checkpoint_best.pth'), name="checkpoint_best.pth", type="model")

        self.on_train_end()
    