import time

from GAN_model import Discriminator, Generator
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transform
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
import matplotlib.pylab as plt
import torch.optim as optim
import os




if __name__=='__main__':
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed_all(1)
    else:
        device = 'cpu'


    trans_dataset = datasets.MNIST('./GAN', train=True, transform=transform.Compose([transform.ToTensor(), transform.Normalize([0.5],[0.5])]), download=True)

    img, label = trans_dataset[0]

    train_dataloader = DataLoader(trans_dataset, batch_size=64, shuffle=True)

    params = {'noise':100, 'size' : (1,28,28)}


    model_g = Generator(params).to(device)
    model_d = Discriminator(params).to(device)


    loss_fn = nn.BCELoss()
    lr = 2e-4
    beta1 = 0.5

    optim_d = optim.Adam(model_d.parameters(), lr, betas=(beta1, 0.999))
    optim_g = optim.Adam(model_g.parameters(), lr, betas=(beta1, 0.999))

    real_label = 1
    fake_label = 0
    nz = params['noise']

    num_epochs = 100

    loss_his = {'g' : [], 'd' : []}

    batch_count = 0
    st_time = time.time()
    model_g.train()
    model_d.train()

    for epoch in range(num_epochs):
        for xb, yb in train_dataloader:

            ba_size = xb.size(0)

            xb = xb.to(device)
            yb_real = torch.Tensor(ba_size, 1).fill_(1.0).to(device)
            yb_fake = torch.Tensor(ba_size, 1).fill_(0.0).to(device)

            # Generator
            model_g.zero_grad()

            noise = torch.randn(ba_size, nz, device=device)
            out_gen = model_g(noise)
            out_dis = model_d(out_gen)

            loss_gen = loss_fn(out_dis, yb_real)
            loss_gen.backward()
            optim_g.step()

            # Discriminator
            model_d.zero_grad()

            out_real = model_d(xb)  # 진짜 이미지 판별
            out_fake = model_d(out_gen.detach())  # 가짜 이미지 판별
            loss_real = loss_fn(out_real, yb_real)
            loss_fake = loss_fn(out_fake, yb_fake)
            loss_dis = (loss_real + loss_fake) / 2

            loss_dis.backward()
            optim_d.step()

            loss_his['g'].append(loss_gen.item())
            loss_his['d'].append(loss_dis.item())

            batch_count += 1
            if batch_count % 1000 == 0:
                print('Epoch: %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' % (
                epoch, loss_gen.item(), loss_dis.item(), (time.time() - st_time) / 60))

    # 가중치 저장
    path2models = './models/'
    os.makedirs(path2models, exist_ok=True)
    path2weights_gen = os.path.join(path2models, 'weights_gen_without_init.pt')
    path2weights_dis = os.path.join(path2models, 'weights_dis_without_init.pt')

    torch.save(model_g.state_dict(), path2weights_gen)
    torch.save(model_d.state_dict(), path2weights_dis)

    weights = torch.load(path2weights_gen)
    model_g.load_state_dict(weights)

    model_g.eval()

    # fake image 생성
    with torch.no_grad():
        fixed_noise = torch.randn(16, 100, device=device)
        img_fake = model_g(fixed_noise).detach().cpu()


    plt.figure(figsize=(10, 10))
    for ii in range(16):
        plt.subplot(4, 4, ii + 1)
        plt.imshow(to_pil_image(0.5 * img_fake[ii] + 0.5), cmap='gray')
        plt.axis('off')

    plt.show()

