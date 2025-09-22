import os
from threading import local

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from utils.utils import get_classes

from .CosineMarginProduct import CosineMarginProduct
from .utils import get_lr

classes_path = "model_data/cls_classes.txt"
class_names, num_classes = get_classes(classes_path)


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        N_rep = x.shape[0]
        N = target.shape[0]
        if not N == N_rep:
            target = target.repeat(N_rep // N, 1)
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class TokenLabelCrossEntropy(nn.Module):
    """
    Token labeling loss.
    """

    def __init__(self, dense_weight=1.0, cls_weight=1.0):
        """
        Constructor Token labeling loss.
        """
        super(TokenLabelCrossEntropy, self).__init__()

        self.dense_weight = dense_weight
        self.cls_weight = cls_weight
        self.CE = SoftTargetCrossEntropy()

    def forward(self, output, aux_output, target):
        target_aux = F.one_hot(target, num_classes=num_classes)
        B, N, C = aux_output.shape
        target_cls = target
        target_aux = target_aux.repeat(1, N).reshape(B * N, C)
        aux_output = aux_output.reshape(-1, C)
        loss_cls = nn.CrossEntropyLoss()(output, target_cls)
        loss_aux = self.CE(aux_output, target_aux)
        return self.cls_weight * loss_cls + self.dense_weight * loss_aux


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss = 0
    total_accuracy = 0

    val_loss = 0
    val_accuracy = 0

    if local_rank == 0:
        print("Start Train")
        pbar = tqdm(total=epoch_step, desc=f"Epoch {epoch + 1}/{Epoch}", postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            outputs, aux_output = model_train(images, targets)
            loss_value = TokenLabelCrossEntropy()(outputs, aux_output, targets)

            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast

            with autocast():
                outputs, aux_output = model_train(images)
                loss_value = TokenLabelCrossEntropy()(outputs, aux_output, targets)
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss_value.item()
        with torch.no_grad():
            accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            total_accuracy += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{"total_loss": total_loss / (iteration + 1), "accuracy": total_accuracy / (iteration + 1), "lr": get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print("Finish Train")
        print("Start Validation")
        pbar = tqdm(total=epoch_step_val, desc=f"Epoch {epoch + 1}/{Epoch}", postfix=dict, mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

            optimizer.zero_grad()

            outputs, aux_output = model_train(images, targets)
            loss_value = nn.CrossEntropyLoss()(outputs, targets)

            val_loss += loss_value.item()
            accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            val_accuracy += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{"total_loss": val_loss / (iteration + 1), "accuracy": val_accuracy / (iteration + 1), "lr": get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print("Finish Validation")
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        print("Epoch:" + str(epoch + 1) + "/" + str(Epoch))
        print("Total Loss: %.3f || Val Loss: %.3f " % (total_loss / epoch_step, val_loss / epoch_step_val))

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print("Save best model to best_epoch_weights.pth")
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
