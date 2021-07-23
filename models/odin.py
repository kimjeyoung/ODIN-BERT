import torch
import math
import copy
from torch import nn
from typing import Optional
from torch import Tensor
from torch.autograd import Variable, grad
from utils import l2_normalize


def BertLinear(i_dim, o_dim, bias=True):
    m = nn.Linear(i_dim, o_dim, bias)
    nn.init.normal_(m.weight, std=0.02)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


class Classifier(nn.Module):
    def __init__(self, backbone,
                 hidden_size=768,
                 num_classes=3,
                 eps=0.002,
                 device='cuda'):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.output_fc = BertLinear(self.hidden_size, self.num_classes)
        self.num_hidden_layers = len(self.backbone.encoder.layer)
        self.device = device
        self.eps = eps

    def get_adv(self, embedding_outputs, loss, use_normalize=False):
        device = self.device
        emb_grad = grad(loss, embedding_outputs, retain_graph=True)

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(emb_grad[0].data, 0)
        gradient = (gradient.float() - 0.5) * 2
        if use_normalize:
            gradient = l2_normalize(gradient)
        p_adv = -1 * self.eps * gradient
        p_adv = Variable(p_adv).to(device)
        return p_adv

    def forward(self, input_ids, token_type_ids: Optional[Tensor] = None,
                attention_mask: Optional[Tensor] = None, adversarial_inputs: Optional[Tensor] = None):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.num_hidden_layers

        embedding_output = self.backbone.embeddings(input_ids, token_type_ids)

        if adversarial_inputs is not None:
            embedding_output += adversarial_inputs

        encoded_layers = self.backbone.encoder(embedding_output,
                                               extended_attention_mask,
                                               head_mask=head_mask)

        sequence_output = encoded_layers[-1]
        cls_feature = sequence_output[:, 0, :]
        cls_output = self.output_fc(cls_feature)
        return cls_output, embedding_output


class Regressor(Classifier):
    def __init__(self, backbone,
                 hidden_size=768,
                 num_classes=3,
                 eps=0.002,
                 device='cpu',
                 val_pos_cons=1e-6):
        super().__init__(backbone, hidden_size, num_classes, eps, device)
        self.output_mean = BertLinear(self.hidden_size, 1)
        self.output_var = BertLinear(self.hidden_size, 1)
        self.gaussian_constant = 0.5 * math.log(2 * math.pi)
        self.var_pos_constant = val_pos_cons

    def regression_nllloss(self, mean, var, y):
        """ Negative log-likelihood loss function. """
        return (torch.log(var) + ((y - mean).pow(2)) / var + self.gaussian_constant).sum()

    def forward(self, input_ids, token_type_ids: Optional[Tensor] = None,
                attention_mask: Optional[Tensor] = None, adversarial_inputs: Optional[Tensor] = None):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.num_hidden_layers

        embedding_output = self.backbone.embeddings(input_ids, token_type_ids)

        if adversarial_inputs is not None:
            embedding_output += adversarial_inputs

        encoded_layers = self.backbone.encoder(embedding_output,
                                               extended_attention_mask,
                                               head_mask=head_mask)

        sequence_output = encoded_layers[-1]
        cls_feature = sequence_output[:, 0, :]
        cls_mean = self.output_mean(cls_feature)
        cls_var = self.output_var(cls_feature)
        output_sig_pos = torch.log(1. + torch.exp(cls_var)) + self.var_pos_constant
        return cls_mean, output_sig_pos, embedding_output


if __name__ == "__main__":
    import numpy as np
    from bert import BertModel, Config

    lm_config = Config()
    bert = BertModel(lm_config)
    bert.load_pretrain_huggingface(torch.load("../ckpt/bert-base-uncased-pytorch_model.bin"))
    model = Regressor(bert)

    input_ids = torch.tensor(np.ones([5, 36])).to(torch.long)
    true = torch.tensor([0, 1, 1, 0, 0]).to(torch.float)
    output_mean, output_var, embedding_output = model(input_ids, input_ids, input_ids)

    loss = model.regression_nllloss(output_mean.view(-1), output_var.view(-1), true.view(-1))
    p_adv = model.get_adv(embedding_output, loss)

    output_mean, output_var, _ = model(input_ids, input_ids, input_ids, p_adv)
    av_loss = model.regression_nllloss(output_mean.view(-1), output_var.view(-1), true.view(-1))

    total_loss = loss + av_loss

    print("TEST")