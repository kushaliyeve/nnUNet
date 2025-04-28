import torch
import wandb
import numpy as np
from typing import List
from torch import distributed as dist
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.wandb_artifact import log_artifact


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
        
    # def on_train_epoch_end(self, train_outputs: List[dict]):
    #     outputs = collate_outputs(train_outputs)

    #     if self.is_ddp:
    #         losses_tr = [None for _ in range(dist.get_world_size())]
    #         dist.all_gather_object(losses_tr, outputs['loss'])
    #         loss_here = np.vstack(losses_tr).mean()
    #     else:
    #         loss_here = np.mean(outputs['loss'])

    #     self.logger.log('train_losses', loss_here, self.current_epoch)
    #     return loss_here
    
    # def on_validation_epoch_end(self, val_outputs: List[dict]):
    #     outputs_collated = collate_outputs(val_outputs)
    #     tp = np.sum(outputs_collated['tp_hard'], 0)
    #     fp = np.sum(outputs_collated['fp_hard'], 0)
    #     fn = np.sum(outputs_collated['fn_hard'], 0)

    #     if self.is_ddp:
    #         world_size = dist.get_world_size()

    #         tps = [None for _ in range(world_size)]
    #         dist.all_gather_object(tps, tp)
    #         tp = np.vstack([i[None] for i in tps]).sum(0)

    #         fps = [None for _ in range(world_size)]
    #         dist.all_gather_object(fps, fp)
    #         fp = np.vstack([i[None] for i in fps]).sum(0)

    #         fns = [None for _ in range(world_size)]
    #         dist.all_gather_object(fns, fn)
    #         fn = np.vstack([i[None] for i in fns]).sum(0)

    #         losses_val = [None for _ in range(world_size)]
    #         dist.all_gather_object(losses_val, outputs_collated['loss'])
    #         loss_here = np.vstack(losses_val).mean()
    #     else:
    #         loss_here = np.mean(outputs_collated['loss'])

    #     global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
    #     mean_fg_dice = np.nanmean(global_dc_per_class)
    #     self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
    #     self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
    #     self.logger.log('val_losses', loss_here, self.current_epoch)
        
    #     return loss_here, global_dc_per_class
        
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
            
            # if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            #     log_artifact()
            wandb.log({
                "epoch": epoch,
                "train_loss": np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4),
                "val_loss": np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4),
                # "pseudo_dice_class_1": [np.round(i, decimals=4) for i in
                #                                self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]]
                # "learning_rate": self.optimizer.param_groups[0]['lr']
            }, step=epoch)

        self.on_train_end()
    