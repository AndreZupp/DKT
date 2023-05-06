import torch.nn as nn
import torch.nn.functional as F


class MultiHeadLoss(nn.Module):
    """Distributed Knowledge Transfer (DKT) Loss function."""

    def __init__(self, model, T=1, use_mse=False, reduction='mean', multi_head_teacher=False, no_kd=False):
        super(MultiHeadLoss, self).__init__()
        self.model = model  # Teacher Model
        self.T = T
        self.mse = use_mse
        self.reduction = reduction
        self.student_loss = None
        self.cl_loss = None
        self.multi_head_teacher = multi_head_teacher
        self.no_kd = no_kd

    def update_mb(self, mb):
        """Updates the saved minibatch
        Args:
            mb (_type_): Minibatch fed to the teacher model.
        """
        self.mb = mb

    def kd_loss(self, student_out):
        student_out = F.log_softmax(student_out / self.T, dim=1)
        teacher_output = self.model(self.mb)
        if self.multi_head_teacher:
            teacher_out = teacher_out[0]
        teacher_out = F.softmax(teacher_output / self.T, dim=1)
        return F.kl_div(student_out, teacher_out, reduction=self.reduction)

    def mse_loss(self, student_out):
        teacher_out = self.model(self.mb)
        if self.multi_head_teacher:
            teacher_out = teacher_out[0]
        return F.mse_loss(student_out, teacher_out, reduction=self.reduction)

    def forward(self, net_out, labels):
        # Net out has 2 output tensors, the first corresponds to the cl head
        # and the second to the student head
        if self.no_kd is False:
            if self.mse:
                self.student_loss = self.mse_loss(net_out[1])
            else:
                self.student_loss = self.kd_loss(net_out[1])
        else:
            self.student_loss = 0
            
        self.cl_loss = F.cross_entropy(net_out[0], labels, reduction=self.reduction)

        return (self.cl_loss, self.student_loss)