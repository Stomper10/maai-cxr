from loadlibs import *
from modules import *
from cfg import configs


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    
    
    train_dataset = ImageDataset(configs, mode='train')
    valid_dataset = ImageDataset(configs, mode='valid')
    datamodule = ImageDataModule(configs=configs, train_dataset=train_dataset, valid_dataset=valid_dataset) 

    model = Model(configs)
    model.train()
    module = CheXpertModule(model=model, configs=configs, lr=configs.learning_rate)

    checkpoints = ModelCheckpoint(
        dirpath=configs.result_dir, 
        monitor="val_acc", 
        mode="min",
        save_top_k=2,
    )
    callbacks = [
        checkpoints, 
        RichProgressBar(), 
        LearningRateMonitor(),
        RichModelSummary(max_depth=3)
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=configs.num_devices, 
        accumulate_grad_batches=1,
        callbacks=callbacks,
        logger=WandbLogger(name=configs.wandb_name, project=configs.wandb_project),
        max_epochs=configs.num_epochs,
        profiler="simple",
        enable_progress_bar=True,
        enable_model_summary=True,
        # precision="16-mixed",
    )

    trainer.fit(
        model=module,
        datamodule=datamodule,
        # ckpt_path=
    )
    torch.distributed.destroy_process_group()