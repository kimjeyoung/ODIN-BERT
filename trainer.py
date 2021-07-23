import torch
import os
import numpy as np
from tqdm import tqdm
from models.bert import BertModel, Config
from models.odin import Classifier, Regressor
from data_loader import DatasetLoader
from torch.utils.data import DataLoader
from models.optimizers import BertAdam
from utils import to_numpy, Accumulator
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, mean_absolute_error


class Trainer:
    def __init__(self, args):
        t_total = -1
        self.epochs = args.epochs
        self.device = args.device
        self.task = args.task
        self.batch_size = args.batch_size
        self.save_path = args.save_path
        self.eps = args.eps
        self.T = args.temperature

        if args.train_or_test == 'train':
            self.train_dataset = DatasetLoader(data_dir=args.data_dir, vocab_path=args.bert_vocab,
                                               max_len=args.max_len, train_or_test='train')
            self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                           drop_last=True)

            self.val_dataset = DatasetLoader(data_dir=args.data_dir, vocab_path=args.bert_vocab,
                                             max_len=args.max_len, train_or_test='val')
            self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
            t_total = len(self.train_loader) * args.epochs

        self.test_dataset = DatasetLoader(data_dir=args.data_dir, vocab_path=args.bert_vocab,
                                          max_len=args.max_len, train_or_test='test')
        self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

        self.num_classes = self.test_dataset.num_classes

        # default config is bert-base
        self.bert_config = Config()
        self.backbone = BertModel(self.bert_config)
        self.backbone.load_pretrain_huggingface(torch.load(args.bert_ckpt))
        if self.task == 'classification':
            self.model = Classifier(self.backbone,
                                    hidden_size=self.bert_config.hidden_size,
                                    num_classes=self.num_classes,
                                    eps=self.eps,
                                    device="cuda" if self.device == 'gpu' else 'cpu')
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.model = Regressor(self.backbone,
                                   hidden_size=self.bert_config.hidden_size,
                                   num_classes=self.num_classes,
                                   eps=self.eps,
                                   device="cuda" if self.device == 'gpu' else 'cpu')
            self.criterion = self.model.regression_nllloss

        if args.device == 'gpu':
            self.model = self.model.to("cuda")
        self.optimizer = BertAdam(self.model.parameters(), lr=args.lr,
                                  warmup=args.warmup, weight_decay=args.weight_decay, t_total=t_total)

        if args.train_or_test == 'test' and os.path.isfile(os.path.join(args.save_path, "bestmodel.bin")):
            self.model.load_state_dict(torch.load(os.path.join(args.save_path, "bestmodel.bin")))

    def train(self):
        best_acc = 0.
        for epoch in range(self.epochs):
            cnt = 0
            self.model.train()
            metrics = Accumulator()
            loader = tqdm(self.train_loader, disable=False)
            loader.set_description('[%s %04d/%04d]' % ('train', epoch, self.epochs))
            for i, batch in enumerate(loader):
                cnt += self.batch_size
                if self.device == 'gpu':
                    batch = [x.to('cuda') for x in batch]
                self.optimizer.zero_grad()
                x_ids, x_segs, x_attns, label = batch
                if self.task == "classification":
                    pred, _ = self.model(x_ids, x_segs, x_attns)
                    loss = self.criterion(pred, label)
                else:
                    pred_mean, pred_var, _ = self.model(x_ids, x_segs, x_attns)
                    loss = self.criterion(pred_mean, pred_var, label)

                metrics.add_dict({
                    'loss': loss.item() * self.batch_size,
                })
                postfix = metrics / cnt
                loader.set_postfix(postfix)
                loss.backward()
                self.optimizer.step()

            val_acc = self.eval()
            if val_acc > best_acc:
                best_acc = val_acc
                test_auroc, test_auprc, test_acc = self.test()
                print(f'\t Test dataset --> AUROC : {test_auroc:.3f} | AUPRC: {test_auprc:.3f} | ACC: {test_acc:.3f}')
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "bestmodel.bin"))

    def eval(self):
        self.model.eval()
        y_true = []
        y_pred = []
        for i, batch in enumerate(self.val_loader):
            if self.device == 'gpu':
                batch = [x.to('cuda') for x in batch]
            x_ids, x_segs, x_attns, label = batch
            pred, _ = self.model(x_ids, x_segs, x_attns)
            if self.task == 'classification':
                pred = to_numpy(torch.argmax(pred, dim=-1)).flatten().tolist()
            else:
                pred = to_numpy(pred).flatten().tolist()
            true = to_numpy(label).flatten().tolist()
            y_true.extend(true)
            y_pred.extend(pred)

        if self.task == 'classification':
            value = accuracy_score(y_true, y_pred)
        else:
            value = 1. / mean_absolute_error(y_true, y_pred)
        return value

    def test(self, training=True):
        self.model.eval()
        y_true = []
        y_preds = []
        ood_preds = []
        for i, batch in enumerate(tqdm(self.test_loader)):
            if self.device == 'gpu':
                batch = [x.to('cuda') for x in batch]
            x_ids, x_segs, x_attns, label = batch

            if self.task == 'classification':
                pred, embedding_output = self.model(x_ids, x_segs, x_attns)

                # temperature scaling
                pred = pred / self.T

                pseudo_label = torch.argmax(pred, dim=-1).to(torch.long)
                loss = self.criterion(pred, pseudo_label)
                p_adv = self.model.get_adv(embedding_output, loss, use_normalize=True)
                pred, _ = self.model(x_ids, x_segs, x_attns, p_adv)

                # temperature scaling
                pred = pred / self.T
                pred = torch.exp(pred) / torch.sum(torch.exp(pred), dim=-1, keepdim=True)

                cls_pred = to_numpy(torch.argmax(pred, dim=-1)).flatten().tolist()
                ood_pred = to_numpy(1. - torch.max(pred, dim=-1)[0]).flatten().tolist()

            else:
                pred_mean, pred_var, embedding_output = self.model(x_ids, x_segs, x_attns)
                loss = self.criterion(pred_mean.view(-1), pred_var.view(-1), label.view(-1).to(torch.float))
                p_adv = self.model.get_adv(embedding_output, loss)
                pred_mean, pred_var, _ = self.model(x_ids, x_segs, x_attns, p_adv)

                cls_pred = to_numpy(pred_mean).flatten().tolist()
                ood_pred = to_numpy(pred_var).flatten().tolist()

            true = to_numpy(label).flatten().tolist()
            y_true.extend(true)
            y_preds.extend(cls_pred)
            ood_preds.extend(ood_pred)

        # ood class idx is 150
        ood_true = (np.array(y_true) == 150).astype(np.uint8).tolist()
        test_auroc = roc_auc_score(ood_true, ood_preds)
        # calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(ood_true, ood_preds)
        test_auprc = auc(recall, precision)

        # calculate accuracy for in-domain
        indomain_true = np.array(y_true)[np.where(np.array(ood_true) == 0)[0]]
        indomain_pred = np.array(y_preds)[np.where(np.array(ood_true) == 0)[0]]
        test_acc = accuracy_score(indomain_true, indomain_pred)
        if training:
            return test_auroc, test_auprc, test_acc
        else:
            print(f'\t Test dataset --> AUROC : {test_auroc:.3f} | AUPRC: {test_auprc:.3f} | ACC: {test_acc:.3f}')

