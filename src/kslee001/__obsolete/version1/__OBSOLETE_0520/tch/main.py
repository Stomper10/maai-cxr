from loadlibs import *
from modules import *
from cfg import configs

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    
    train_dataset = ImageDataset(configs, mode='train')
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=configs.batch_size,
        num_workers=configs.num_workers,
        shuffle=True
    )
    valid_dataset = ImageDataset(configs, mode='valid')
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=configs.batch_size,
        num_workers=configs.num_workers,
        shuffle=False
    )
    
    model = Model(configs)
    model = model.cuda()
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=configs.learning_rate,
    )
    criterion = torch.nn.BCEWithLogitsLoss().cuda()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=configs.num_epochs, eta_min=0)

    acc_meter = BinaryAccuracy()

    print("[INFO] begin training...")
    # epoch
    for epoch in range(configs.num_epochs):
        epoch_loss = 0.0

        train_ypred_score = 0
        train_ygt_score = 0
        valid_ypred_score = 0
        valid_ygt_score = 0
        
        model.train()
        start_time = time.time()
        # step
        for idx, batch in (enumerate(train_loader)):
            optimizer.zero_grad()
            x, y = batch
            x, y = x.cuda(), y.cuda()
            yhat = model(x)  # use teacher forcing during training
            
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*y.numel()

            # print ETA every batch
            elapsed_time = time.time() - start_time
            elapsed_time_per_step = elapsed_time / (idx + 1)
            eta_dt = timedelta(seconds=int(elapsed_time_per_step * (len(train_loader) - idx - 1)))
            eta = str(eta_dt)
            spent_dt = timedelta(seconds=int(elapsed_time))   
            spent = str(spent_dt)

            with torch.no_grad():
                yhat = torch.nn.functional.sigmoid(yhat).detach().cpu().round()
                y = y.detach().cpu()

            ypred_score = torch.sum(torch.sum(yhat==y))
            train_ypred_score += ypred_score
            train_ygt_score += len(y)
            step_acc = ypred_score/len(y)

            print(f"""--current batch {idx*configs.batch_size}/{len(train_loader)*configs.batch_size} | loss : {loss.item():.4f} | acc  : {step_acc:.4f} | time : {spent}-{str(spent_dt+eta_dt)} | eta : {eta} """, end='\r')

        train_loss = epoch_loss / configs.num_epochs
        train_acc = train_ypred_score / train_ygt_score
        
        elapsed_time = time.time() - start_time
        eta = str(timedelta(seconds=int(elapsed_time * (configs.num_epochs - epoch - 1))))
        if scheduler is not None:
            scheduler.step()

        # eval
        model.eval()
        with torch.no_grad():
            total_loss = 0.0

            
            for idx, batch in enumerate(valid_loader):
                x, y = batch
                x, y = x.cuda(), y.cuda()
                yhat = model(x)
                loss = criterion(yhat, y)
                total_loss += loss.item()*y.numel()
                
                yhat = torch.nn.functional.sigmoid(yhat).detach().cpu().round()
                y = y.detach().cpu()
                
                ypred_score = torch.sum(torch.sum(yhat==y))
                valid_ypred_score += ypred_score
                valid_ygt_score += len(y)
                # step_acc = ypred_score/len(y)

        valid_loss = total_loss / configs.num_epochs
        valid_acc = valid_ypred_score / valid_ygt_score
            
        print(f"\n[INFO] Epoch {epoch+1} | ETA : {eta}")
        print(f"-- train loss : {train_loss:.4f} | train acc : {train_acc:.4f}")
        print(f"-- valid loss : {valid_loss:.4f} | valid acc : {valid_acc:.4f}")
        
        
    elapsed_time = time.time() - start_time

    # Convert the elapsed time to minutes and seconds
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = int(elapsed_time % 60)
        

    print(f"[INFO] training done | elapsed time : {elapsed_minutes} min {elapsed_seconds} sec")
