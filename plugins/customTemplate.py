from avalanche.training import Naive


class CustomNaive(Naive):

    def __init__(self, model: "Module", optimizer: "Optimizer", criterion=..., train_mb_size: int = 1, train_epochs: int = 1, eval_mb_size: "Optional[int]" = None, device=None, plugins: "Optional[List[SupervisedPlugin]]" = None, evaluator: "EvaluationPlugin" = ..., eval_every=-1, kd_rel = 0.5, **base_kwargs):
        super().__init__(model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, device, plugins, evaluator, eval_every, **base_kwargs)
        self.kd_rel = kd_rel

    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)
            
            # Loss & Backward
            self.loss += (self.criterion()[0] + self.kd_rel * self.criterion()[1])

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)