import time
import torch
import numpy as np
import matplotlib.pyplot as plt

def set_figsize(figsize=(3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figsize

def evaluate_accuracy(data_iter, net, loss, device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            net.eval()
            y_hat = net(X)
            l = loss(y_hat, y.long())
            acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
            net.train()
    return acc_sum / n, l.item()

def train(net, train_iter, valida_iter, loss, optimizer, device, datasets, image_path, iter_index, epochs=30, early_stopping=True, early_num=20):
    model_path = image_path[0:-7] + 'models/early.pth'
    
    early_epoch = 0
    net.to(device)
    print(f"Training on {device}")
    start = time.time()
    
    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []
    
    lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
    
    for epoch in range(epochs):
        train_acc_sum, train_l_sum, total_batches = 0.0, 0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            
            y_hat = net(X)
            l = loss(y_hat, y.long())
            
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            total_batches += 1
        
        lr_adjust.step()
        
        valida_acc, valida_loss = evaluate_accuracy(valida_iter, net, loss, device)
        
        train_loss_list.append(train_l_sum / total_batches)
        train_acc_list.append(train_acc_sum / len(train_iter.dataset))
        
        valida_loss_list.append(valida_loss)
        valida_acc_list.append(valida_acc)
        
        print(f"Epoch {epoch + 1}, Train Loss: {train_l_sum / total_batches:.6f}, Train Acc: {train_acc_sum / len(train_iter.dataset):.3f}, "
              f"Valida Loss: {valida_loss:.6f}, Valida Acc: {valida_acc:.3f}")
        
        if early_stopping:
            if valida_loss_list[-1] > min(valida_loss_list):
                early_epoch += 1
                if early_epoch == early_num:
                    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            else:
                early_epoch = 0
                torch.save(net.state_dict(), model_path)
    
    set_figsize()
    _, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    axs[0, 0].plot(np.linspace(1, epoch + 1, len(train_acc_list)), train_acc_list, color='green')
    axs[0, 0].set_title('Train Accuracy')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Accuracy')
    
    axs[0, 1].plot(np.linspace(1, epoch + 1, len(valida_acc_list)), valida_acc_list, color='deepskyblue')
    axs[0, 1].set_title('Valida Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    
    axs[1, 0].plot(np.linspace(1, epoch + 1, len(train_loss_list)), train_loss_list, color='red')
    axs[1, 0].set_title('Train Loss')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss')
    
    axs[1, 1].plot(np.linspace(1, epoch + 1, len(valida_loss_list)), valida_loss_list, color='gold')
    axs[1, 1].set_title('Valida Loss')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Loss')
    
    plt.tight_layout()
    
    date = time.strftime('%Y-%m-%d-%H:%M', time.localtime())
    plt.savefig(f'{image_path}/{iter_index}-{datasets}-{date}.png')
    print(f"Epoch {epoch + 1}, Final Train Acc: {train_acc_list[-1]:.3f}, Final Valida Acc: {valida_acc_list[-1]:.3f}, "
          f"Figures saved at {image_path}")