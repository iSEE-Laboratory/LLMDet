from mmengine.hooks import Hook
from mmengine.runner import Runner

from typing import Optional

from mmdet.registry import HOOKS

@HOOKS.register_module()
class GradientCheckHook(Hook):
    def after_train_iter(self,
            runner: Runner,
            batch_idx: int,
            data_batch: Optional[dict] = None,
            outputs: Optional[dict] = None) -> None:
        # 获取模型
        model = runner.model
        # 遍历模型的参数，打印梯度
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                print(f"{name} gradient norm: {param.grad.norm()}")
