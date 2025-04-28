import torch
import wandb
import numpy as np
from time import time
from batchgenerators.utilities.file_and_folder_operations import join


from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
class nnUNetTrainer_Wandb_Logger(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)

        wandb.init(
                project="pinkcc",
                name=f"Dataset_001_PINKCC_{self.fold}",
                config={
                    "plans": self.plans_manager,
                    "configuration": self.configuration_manager,
                    "fold": self.fold,
                    "num_epochs": self.num_epochs
                }
            )
        
    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        dice_scores = self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]
        wandb.log({
            "epoch": self.current_epoch,
            "train_loss": np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4),
            "val_loss": np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4),
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            **{f'pseudo_dice_class_{class_idx}': np.round(dice_value, decimals=4)
               for class_idx, dice_value in enumerate(dice_scores)}
        }, step=self.current_epoch)
        
        
        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))
            wandb.log({
                "best_ema": np.round(self._best_ema, decimals=4),
                "best_epoch": self.current_epoch
            }, step=self.current_epoch)
            
            artifact = wandb.Artifact('best_model', type='model')
            artifact.add_file(join(self.output_folder, 'checkpoint_best.pth'))
            wandb.log_artifact(artifact, aliases=['latest', 'best'])
            

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1
        
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

        self.on_train_end()
    