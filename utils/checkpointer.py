from typing import Union, Dict, Any
from collections import OrderedDict
import logging
import os

import torch

logger = logging.getLogger(__name__)


class Checkpointer:
    def __init__(self, serialization_dir: str="output", exclude_prefix: str=None) -> None:
        self._serialization_dir = serialization_dir
        if not os.path.exists(self._serialization_dir):
            os.makedirs(self._serialization_dir, exist_ok=True)

        self.exclude_prefix = exclude_prefix

    def auto_save_checkpoint(self, model, config, current_epoch, global_step, optimizer, scheduler, accelerator=None, 
            epoch_flag=False, step_flag=False, only_latest=False, extra_training_states={}):

        if not only_latest and 'ckpt_frequent' in config:
            # we may not save the checkpoint every epoch
            epoch_flag = epoch_flag and (current_epoch+1) % config['ckpt_frequent'] == 0
        
        if step_flag or epoch_flag or only_latest:
            model_without_ddp = model
            if hasattr(model, 'module'):
                model_without_ddp = model.module

            model_state_dict = model_without_ddp.state_dict()
            if self.exclude_prefix is not None:
                # we do not save those params named with the specified prefix
                model_state_dict = OrderedDict({k: v for k, v in model_state_dict.items() if not k.startswith(self.exclude_prefix)})

            save_obj = {
                'model': model_state_dict,
                'config': config,
                'epoch': current_epoch,
                'step': global_step,
            }
            
            training_states = {
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'amp': accelerator.state_dict() if accelerator is not None else None,
                'epoch': current_epoch,
                'step': global_step,
                **extra_training_states,
            }

            self.save_checkpoint(
                model_state=save_obj, 
                epoch=current_epoch, 
                step=global_step if step_flag else -1, 
                training_states=training_states,
                only_latest=only_latest,
            )
    
    def get_latest_states_paths(self):
        model_state_path = os.path.join(self._serialization_dir, 'model_state_latest.th')
        training_states_path = os.path.join(self._serialization_dir, 'training_state_latest.th')
        return model_state_path, training_states_path

    def save_checkpoint(self,
                        epoch: Union[int, str],
                        model_state: Dict[str, Any],
                        training_states: Dict[str, Any],
                        step: int = -1,
                        only_latest: bool = False) -> None:
        """
        Save ckpt to local or HDFS
        """
        model_state_path, training_states_path = self.get_latest_states_paths()
        torch.save(model_state, model_state_path)
        torch.save(training_states, training_states_path)
        
        if not only_latest:
            fn = f"model_state_step_{step}.th" if step > 0 else f"model_state_epoch_{epoch}.th"
            model_path = os.path.join(self._serialization_dir, fn)
            torch.save(model_state, model_path)

    def resume_latest_states(self, model, optimizer, lr_scheduler, accelerator=None, return_type='step'):
        model_state_path, training_states_path = self.get_latest_states_paths()

        assert os.path.exists(model_state_path) == os.path.exists(training_states_path)

        if not os.path.exists(model_state_path):
            if return_type in ['step', 'epoch']:
                return 0
            return None
        
        print('Resuming')
        print('### Loading model\'s state from', model_state_path)
        checkpoint = torch.load(model_state_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        if all(['module.' in item for item in model.state_dict()]):
            state_dict = {'module.%s'%k: v for k, v in state_dict.items()}
        
        msg = model.load_pretrained_state_dict(state_dict)
        assert len(msg.unexpected_keys) == 0, msg.unexpected_keys
        for key in msg.missing_keys:
            assert self.exclude_prefix is not None
            assert key.startswith(f'{self.exclude_prefix}') or key.startswith(f'module.{self.exclude_prefix}'), key
            # assert getattr(model, key).requires_grad is False, key

        print('### Loading training states from', training_states_path)
        training_states = torch.load(training_states_path, map_location='cpu') 
 
        optimizer.load_state_dict(training_states['optimizer'])
        lr_scheduler.load_state_dict(training_states['lr_scheduler'])
        if accelerator is not None:
            accelerator.load_state_dict(training_states['amp'])

        if return_type == 'step':
            return training_states['step'] + 1
        elif return_type == 'epoch':
            return training_states['epoch'] + 1
        else:
            return training_states
