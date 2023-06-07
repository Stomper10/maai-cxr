from loadlibs import *


class ImageDataset(torch.utils.data.Dataset): # batch dataset
    def __init__(self, configs, mode='train', augment=False):
        # arguments
        self.configs = configs
        self.image_size = configs.image_size
        self.image_add_size = configs.image_add_size
        self.mode = mode
        self.augment = augment
        self.data_dir = f"{configs.dataset_dir}/{mode}.csv"
        
        # configuration
        self.batch_size = configs.batch_size
        self.image_size = configs.image_size
        
        # data processing    
        data = pd.read_csv(self.data_dir).fillna(0.0)
        data['Path'] = data['Path'].str.replace(configs.dataset_name, configs.dataset_dir, regex=False)
        for col_idx in range(5, len(data.columns)):
            data[data.columns[col_idx]] = data[data.columns[col_idx]].astype(str)
            data[data.columns[col_idx]] = data[data.columns[col_idx]].str.replace("-1.0", "0.0").astype(float).fillna(0.0)
            # data[data.columns[col_idx]] = data[data.columns[col_idx]].astype(int)
        
        # store data
        self.X = data['Path'].values
        self.Y = data.values[:, 5:].astype(np.int64)

        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.resize(cv2.imread(self.X[idx]))
        if self.augment:
            x = self.train_transform(x)
        x = torch.from_numpy(x).permute(2,0,1).float()
        
        y = (torch.from_numpy(self.Y[idx]).float()
             if (self.mode in ['train', 'valid', 'val'] )
             else None
            )
        return x, y
    
    def resize(self, x):
        H, W, C = x.shape
        
        if H>W:
            resize_fn = A.Resize(
                height=self.image_size[0]*H//W + self.image_add_size, 
                width=self.image_size[1] + self.image_add_size
            )
        else: # W < H
            resize_fn = A.Resize(
                height=self.image_size[0] + self.image_add_size, 
                width=self.image_size[1]*W//H + self.image_add_size
            )
        
        x = A.Compose([
            resize_fn,
            A.CenterCrop(height=self.image_size[0], width=self.image_size[1]),            
        ])(image=x)['image']
        
        return x     # dtype : numpy array    
    
    def train_transform(self, x):
        x = A.Compose([
            A.Affine(
                scale = 0.95,
                translate_percent=0.05, # moving
                shear = 0.05,           # distortion      
                rotate=None,
                interpolation=1,
            ),
        ])(image=x)['image']
        return x


# =======================================================================
class ImageDataModule(pl.LightningDataModule):
    def __init__(self, configs, train_dataset, valid_dataset):
        super().__init__()
        self.configs = configs

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.configs.batch_size,
            num_workers=self.configs.num_workers,
            # pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.valid_dataset,
            num_workers=self.configs.num_workers,
            batch_size=self.configs.batch_size,
            # pin_memory=True,
            drop_last=True,
            shuffle=False,
        )

# ======================================================================
class CheXpertModule(pl.LightningModule):
    def __init__(self, model, configs, lr):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.configs = configs
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = MultilabelAccuracy(14)
        self.val_acc = MultilabelAccuracy(14)

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(
        #     self.model.parameters(), lr=self.hparams.lr, weight_decay=0.01
        # )
        optimizer = create_optimizer_v2(
            self.model, "madgradw", lr=self.hparams.lr, weight_decay=self.configs.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
        }

        return [optimizer], [scheduler_config]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)        
        
        loss = self.criterion(yhat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.train_acc(yhat, y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        
        loss = self.criterion(yhat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_acc(yhat, y)
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        

# ==================================================
    class Model(torch.nn.Module):
        def __init__(self, configs):
            super().__init__()
            self.backbone = create_model(
                configs.backbone, 
                pretrained=True, 
                num_classes=0,
            )
            self.linear = torch.nn.Linear(configs.feature_dim, configs.num_classes)
        def forward(self, x):
            return self.linear(self.backbone(x))