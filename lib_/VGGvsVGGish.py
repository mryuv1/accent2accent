import torch
import pytorch_lightning as pl
from vgg2 import VGGEncoder
from DataLoader_VGG import AccentHuggingBasedDataLoader

class VGGvsVGGish(pl.LightningModule):
    def __init__(self, VGGish, lr, b1, b2):
        super().__init__()
        self.save_hyperparameters()
        self.model = VGGEncoder(VGGish=VGGish, TzlilTrain=True, num_classes=6)
        self.loss = torch.nn.CrossEntropyLoss()
        self.cntr_loss = 0
        self.step_cntr = 0
        self.acc_cntr = 0
        self.VGGish = VGGish

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(b1, b2))
        return optimizer

    def forward(self, xs):
        return self.model(xs)

    def training_step(self, batch, batch_idx):
        self.step_cntr += 1
        x, y = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        if self.VGGish:
            x = x[:,0:1,:,:]
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.cntr_loss += loss
        self.log('train_loss', loss)
        if self.VGGish:
            wandb.log({"VGGish_train_loss": loss})
        else:
            wandb.log({"vgg_train_loss": loss})
        #Calculate the accuracy
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).item() / (len(y) * 1.0)
        self.log('train_acc', acc)
        if self.VGGish:
            wandb.log({"VGGish_train_acc": acc})
        else:
            wandb.log({"VGGtrain_acc": acc})
        #wandb.log({"train_acc": acc})
        self.acc_cntr += acc
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).item() / (len(y) * 1.0)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def on_validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)
        self.log('avg_val_acc', avg_acc)

    def on_train_batch_end(self, outputs, a=0,b=0,c=0,**_):
        #Check if the iteration is divisible by 10
        if self.step_cntr % 10 == 0:
            self.log('avg_train_loss', self.cntr_loss / 10)
            print("Iteration: ", self.step_cntr, "Loss: ", self.cntr_loss / 10)
            self.cntr_loss = 0
            self.log('avg_train_acc', self.acc_cntr / 10)
            print("Iteration: ", self.step_cntr, "Accuracy: ", self.acc_cntr / 10)
            self.acc_cntr = 0
def train_model(model, dataloader, max_epochs):
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(model, dataloader)

def run_vgg_vs_vggish():
    # Initialize data loaders
    data_module_vgg = AccentHuggingBasedDataLoader(batch_size=32, SlowRun=False, TzlilTrain=True)
    data_module_vggish = AccentHuggingBasedDataLoader(batch_size=32,  SlowRun=False, TzlilTrain=True)

    # Initialize models
    vgg_model = VGGvsVGGish(VGGish=False, lr=0.001, b1=0.9, b2=0.999)
    vggish_model = VGGvsVGGish(VGGish=True, lr=0.001, b1=0.9, b2=0.999)

    if torch.cuda.is_available():
        vgg_model = vgg_model.cuda()
        vggish_model = vggish_model.cuda()



    # Train VGGish model
    print("Training VGGish Model...")
    train_model(vggish_model, data_module_vggish.train_dataloader(), max_epochs=10)

    # Train VGG model
    print("Training VGG Model...")
    train_model(vgg_model, data_module_vgg.train_dataloader(), max_epochs=10)


if __name__ == "__main__":
    import wandb
    wandb.init(project="AdaCONV")
    run_vgg_vs_vggish()

# import torch
# import pytorch_lightning as pl
# from lib_.vgg2 import VGGEncoder
# from lib_.DataLoader_VGG import AccentHuggingBasedDataLoader
#
#
# # import wandb
# class VGGvsVGGish(pl.LightningModule):
#     def __init__(self, VGGish, lr, b1, b2):
#         super().__init__()
#         self.save_hyperparameters()
#         self.model = VGGEncoder(VGGish=VGGish, TzlilTrain=True, num_classes=6)
#         self.loss = torch.nn.CrossEntropyLoss()
#         self.step_cntr = 0
#         self.acc_cntr = 0
#         self.VGGish = VGGish
#
#     def configure_optimizers(self):
#         lr = self.hparams.lr
#         b1 = self.hparams.b1
#         b2 = self.hparams.b2
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(b1, b2))
#         return optimizer
#
#     def forward(self, xs):
#         return self.model(xs)
#
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         print("X shape: ", x.shape)
#         print("Y shape: ", y.shape)
#         x = x.squeeze(0)
#         y = y.squeeze(0)
#         print(x.shape)
#         print(y.shape)
#         logits = self.forward(x)
#         loss = self.loss(logits, y)
#         self.log('train_loss', loss)
#         # Calculate the accuracy
#         preds = torch.argmax(logits, dim=1)
#         acc = torch.sum(preds == y).item() / (len(y) * 1.0)
#         self.log('train_acc', acc)
#         self.acc_cntr += acc
#         return loss, acc
#
#
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self.forward(x)
#         loss = self.loss(logits, y)
#         self.log('val_loss', loss)
#         return loss
#
#     def on_validation_epoch_end(self, outputs):
#         avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
#         self.log('avg_val_loss', avg_loss)
#
#
# class CombinedModel(pl.LightningModule):
#     def __init__(self, vgg_model, vggish_model):
#         super().__init__()
#         self.vgg_model = vgg_model
#         self.vggish_model = vggish_model
#         self.vgg_loss_cntr = 0
#         self.vggish_loss_cntr = 0
#         self.vgg_acc_cntr = 0
#         self.vggish_acc_cntr = 0
#         self.step_cntr = 0
#         self.automatic_optimization = False
#
#     def forward(self, xs):
#         return self.vgg_model(xs), self.vggish_model(xs)
#
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         x = x.squeeze(0)
#         y = y.squeeze(0)
#         vgg_out, vggish_out = self.forward(x)
#         vgg_loss, vgg_acc = self.vgg_model.training_step((vgg_out, y), batch_idx)
#         vggish_loss, vggish_acc = self.vggish_model.training_step((vggish_out, y), batch_idx)
#         self.vgg_loss_cntr += vgg_loss
#         self.vggish_loss_cntr += vggish_loss
#         self.vgg_acc_cntr += vgg_acc
#         self.vggish_acc_cntr += vggish_acc
#         if self.step_cntr % 5 == 0:
#             self.log('avg_vgg_loss', self.vgg_loss_cntr / 5)
#             self.log('avg_vggish_loss', self.vggish_loss_cntr / 5)
#             self.log('avg_vgg_acc', self.vgg_model.acc_cntr / 5)
#             self.log('avg_vggish_acc', self.vggish_model.acc_cntr / 5)
#             if self.VGGish:
#                 wandb.log({"VGGish_train_acc": self.vgg_acc_cntr / 5})
#                 wandb.log({"VGGish_train_loss": self.vggish_loss_cntr / 5})
#             else:
#                 wandb.log({"VGG_train_acc": self.vgg_acc_cntr / 5})
#                 wandb.log({"VGG_train_loss": self.vgg_loss_cntr / 5})
#
#             self.vgg_loss_cntr = 0
#             self.vgg_model.acc_cntr = 0
#             self.vggish_loss_cntr = 0
#             self.vggish_model.acc_cntr = 0
#             print("Iteration: ", self.step_cntr, "VGG Loss: ", self.vgg_loss_cntr / 5, "VGGish Loss: ",
#                   self.vggish_loss_cntr / 5, "VGG Acc: ", self.vgg_model.acc_cntr / 5, "VGGish Acc: ",
#                   self.vggish_model.acc_cntr / 5)
#         self.step_cntr += 1
#         return vgg_loss + vggish_loss
#
#     # def validation_step(self, batch, batch_idx):
#     #     x, y = batch
#     #     vgg_out, vggish_out = self.forward(x)
#     #
#     #     vgg_loss = self.vgg_model.validation_step((vgg_out, y), batch_idx)
#     #     vggish_loss = self.vggish_model.validation_step((vggish_out, y), batch_idx)
#     #
#     #     return vgg_loss + vggish_loss
#
#     def configure_optimizers(self):
#         vgg_optimizer = self.vgg_model.configure_optimizers()
#         vggish_optimizer = self.vggish_model.configure_optimizers()
#
#         return vgg_optimizer, vggish_optimizer
#
#
# def train_model(model, dataloader, max_epochs):
#     trainer = pl.Trainer(max_epochs=max_epochs, accelerator="cpu")
#     trainer.fit(model, dataloader)
#
#
# def run_vgg_vs_vggish():
#     # Initialize data loaders
#     data_module_combined = AccentHuggingBasedDataLoader(batch_size=32, SlowRun=False, TzlilTrain=True)
#
#     # Initialize models
#     vgg_model = VGGvsVGGish(VGGish=False, lr=0.001, b1=0.9, b2=0.999)
#     vggish_model = VGGvsVGGish(VGGish=True, lr=0.001, b1=0.9, b2=0.999)
#     # wandb.watch(vgg_model)
#     # wandb.watch(vggish_model)
#
#     combined_model = CombinedModel(vgg_model, vggish_model)
#
#     if torch.cuda.is_available():
#         combined_model = combined_model.cuda()
#
#     # Train VGG and VGGish models together
#     print("Training VGG and VGGish Models together...")
#     train_model(combined_model, data_module_combined.train_dataloader(), max_epochs=10)
#
#
# if __name__ == "__main__":
#     # wandb.init(project="AdaCONV")
#     import wandb
#     wandb.init(project="AdaCONV")
#     run_vgg_vs_vggish()
#
