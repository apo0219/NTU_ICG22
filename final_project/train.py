from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn
from dataset import DataSet, StyleSet
from torchvision import transforms
import argparse
from myclass import StyleBank, MyLoss, MyConv, DenseBlock, Paper_StyleBank
from PIL import Image
from glob import glob
import os
from torchvision.utils import save_image


class Trainer():
    def __init__(self, args, model, style_images, optimizer_auto, optimizer_style, loss):
        # load dataset
        self.args = args 
        self.style_images = style_images
        self.model = model
        self.optimizer_auto = optimizer_auto
        self.optimizer_style = optimizer_style
        self.loss = loss
    

    def fit(self, train_dataset, test_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        print('[>] DataLoader done')
        # for i in range(len(self.model.bank)):
        #   self.model.bank[i] = self.model.bank[i].to(self.args.device)
        self.model = self.model.to(self.args.device)
        style_counter = 0
        self.loss = self.loss.to(self.args.device)

        for epoch in range(self.args.epoch):
            total_org_loss = 0
            total_style_loss = 0
            print(f'[>] epoch:{epoch+1}')
            for i, x in tqdm(enumerate(train_loader)):
                x = x.to(self.args.device)
                if i % (self.args.t+1) == self.args.t: # do loss origin
                    out_org = self.model(x)
                    loss_ = self.loss(out_org, x, None, mode = 'auto')
                    total_org_loss += loss_.item()
                    lastx = x

                    self.optimizer_auto.zero_grad()
                    loss_.backward()
                    self.optimizer_auto.step()
                else:
                    out_style = self.model(x, style_counter)
                    loss_ = self.loss(out_style, x, self.style_images[style_counter].to(self.args.device), mode = 'style')
                    total_style_loss += loss_.item()
                    style_counter += 1
                    style_counter %= len(self.style_images)
                    lastx = x

                    self.optimizer_style.zero_grad()
                    loss_.backward()
                    self.optimizer_style.step()


            if epoch % self.args.save_epoch == (self.args.save_epoch - 1):
                checkpoint = {'model':self.model, 'optimizer_auto':self.optimizer_auto, 'optimizer_style':self.optimizer_style}
                if not os.path.exists(self.args.output_dir):
                    os.mkdir(self.args.output_dir)
                epoch_dir = os.path.join(self.args.output_dir, f'epoch{epoch}')
                if not os.path.exists(epoch_dir):
                    os.mkdir(epoch_dir)
                torch.save(checkpoint, os.path.join(epoch_dir , f'checkpoint.pth'))
                with torch.no_grad():
                    out_org = self.model(lastx)
                    out_styles = [None] * len(self.style_images)
                    for i in range(len(self.style_images)):
                        out_styles[i] = self.model(lastx, i)
                    
                    for i in range(len(out_org)):
                        org = out_org[i]
                        org = org.cpu()
                        save_image(org, os.path.join(epoch_dir, f'auto_enc{i}.jpg'))

                    for i in range(len(lastx)):
                        lastxorg = lastx[i]
                        lastxorg = lastxorg.cpu()
                        save_image(lastxorg, os.path.join(epoch_dir, f'origin{i}.jpg'))

                    for si in range(len(out_styles)):
                        for i in range(len(out_styles[si])):
                            out_style_ = out_styles[si][i]
                            out_style_ = out_style_.cpu()
                            save_image(out_style_, os.path.join(epoch_dir, f'image{i}_style{si}.jpg'))
                    for i in range(len(self.style_images)):
                        lastxstyle = self.style_images[i]
                        save_image(lastxstyle, os.path.join(epoch_dir, f'origin_style{i}.jpg'))
            
            print(f'total org loss @ epoch {epoch} : {total_org_loss/len(train_loader)}')
            print(f'total style loss @ epoch {epoch} : {total_style_loss/len(train_loader)}')

            if epoch % self.args.eval_epoch == (self.args.eval_epoch - 1):
                self.eval(test_dataset)

    def eval(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)
        total_style_loss = 0
        total_origin_loss = 0
        with torch.no_grad():
            for i, x in tqdm(enumerate(test_loader)):
                x = x.to(self.args.device)
                out_org = self.model(x)
                loss_origin = self.loss(out_org, x, None, mode = 'auto')
                for si in range(len(self.style_images)):
                    out_style = self.model(x, si)
                    loss_style = self.loss(out_style, x, self.style_images[si].to(self.args.device), mode = 'style')
                    total_style_loss += loss_style.item()/len(self.style_images)
                total_origin_loss += loss_origin

            total_style_loss /= len(test_loader)
            total_origin_loss /= len(test_loader)

        # if not os.path.exists(save_dir):
        #   os.mkdir(save_dir)
        # record = torch.stack(test_dataset[:5])
        # with torch.no_grad():
        #   record = record.to(self.args.device)
        #   out_org = self.model(record)
        #   for i in range(len(out_org)):
        #       org = torch.permute(out_org[i], (1,2,0))
        #       save_image(org, os.path.join(save_dir, 'org{i}.jpg'))
        #   for si in range(len(self.style_images)):
        #       out_style = self.model(x, si)
        #       for i in range(len(out_style)):
        #           style = torch.permute(out_style[i], (1,2,0))
        #           save_image(style, os.path.join(save_dir, 'img{i}_style{si}.jpg'))
        
        print(f'{total_style_loss=}, {total_origin_loss=}')
        return total_style_loss, total_origin_loss

    def trian_new_style(self, train_dataset, test_dataset, path):
        train_style_bank_idx = len(self.model.bank)
        train_idx = self.style_images.images.index(path)
        assert train_idx != -1,"train idx not OK"
        print(f'{train_idx=}')
        self.model.bank.append( 
            nn.Sequential(
                MyConv( 256, 128, 3, 1, 1, padding_mode='zeros' ),
                MyConv( 128, 128, 3, 1, 1, padding_mode='zeros' ),
                MyConv( 128, 256, 3, 1, 1, padding_mode='zeros' )
            )
            # nn.Sequential(
            #     DenseBlock( 256, 3, 64 ), # 256 + 3 * 64 = 448
            #     nn.Conv2d( 448, 256, kernel_size = 1, padding = 0, stride = 1 ),
            #     nn.InstanceNorm2d( 256 ),
            #     nn.ReLU()
            # )
        )

        self.optimizer_style = AdamW(self.model.bank.parameters(), lr=self.args.learning_rate)
        self.model.encoder.eval()
        self.model.decoder.eval()
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        print('[>] DataLoader done')
        self.model = self.model.to(self.args.device)
        self.loss = self.loss.to(self.args.device)
        

        for epoch in range(self.args.epoch):
            total_style_loss = 0
            print(f'[>] epoch:{epoch+1}')
            for i, x in tqdm(enumerate(train_loader)):
                x = x.to(self.args.device)
                out_style = self.model(x, train_style_bank_idx)
                loss_ = self.loss(out_style, x, self.style_images[train_idx].to(self.args.device), mode = 'style')
                total_style_loss += loss_.item()
                lastx = x
                self.optimizer_style.zero_grad()
                loss_.backward()
                self.optimizer_style.step()


            if epoch % self.args.save_epoch == (self.args.save_epoch - 1):
                checkpoint = {'model':self.model, 'optimizer_auto':self.optimizer_auto, 'optimizer_style':self.optimizer_style}
                if not os.path.exists(self.args.output_dir):
                    os.mkdir(self.args.output_dir)
                epoch_dir = os.path.join(self.args.output_dir, f'epoch{epoch}')
                if not os.path.exists(epoch_dir):
                    os.mkdir(epoch_dir)
                torch.save(checkpoint, os.path.join(epoch_dir , f'checkpoint.pth'))
                with torch.no_grad():
                    out_x = self.model(lastx)
                    out_style = self.model(lastx, train_style_bank_idx)

                    
                    for si in range(len(out_style)):
                        out_style_ = out_style[si].cpu()
                        save_image(out_style_, os.path.join(epoch_dir, f'image{si}_with_style.jpg'))
                    
                    for i in range(len(lastx)):
                        lastxorg = lastx[i]
                        lastxorg = lastxorg.cpu()
                        save_image(lastxorg, os.path.join(epoch_dir, f'origin{i}.jpg'))

                    for i in range(len(out_x)):
                        outx_ = out_x[i]
                        outx_ = outx_.cpu()
                        save_image(outx_, os.path.join(epoch_dir, f'auto{i}.jpg'))

                    lastxstyle = self.style_images[train_idx]
                    save_image(lastxstyle, os.path.join(epoch_dir, f'origin_style.jpg'))
            
            print(f'total style loss @ epoch {epoch} : {total_style_loss/len(train_loader)}')

    def predict(self, image_set):
        if not os.path.exists(self.args.output_dir):
            os.mkdir(self.args.output_dir)
        img_loader = DataLoader(image_set, batch_size=self.args.batch_size, shuffle=False)
        self.model.encoder.eval()
        self.model.decoder.eval()
        with torch.no_grad():
            self.model = self.model.to(self.args.device)
            for i, x in tqdm(enumerate(img_loader)):
                x = x.to(self.args.device)
                out_org = self.model(x)
                for idx in range(len(out_org)):
                    out_org_ = out_org[idx]
                    out_org_ = out_org_.cpu()
                    save_image(out_org_, os.path.join(self.args.output_dir, f'batch{i}_image{idx}_auto.jpg'))

                for idx in range(len(x)):
                    x_ = x[idx]
                    x_ = x_.cpu()
                    save_image(x_, os.path.join(self.args.output_dir, f'batch{i}_image{idx}_origin.jpg'))

                for si in range(len(self.style_images)):
                    out_style = self.model(x, si)
                    for sj in range(len(out_style)):
                        out_style_ = out_style[sj]
                        out_style_ = out_style_.cpu()
                        save_image(out_style_, os.path.join(self.args.output_dir, f'batch{i}_image{sj}_style{si}.jpg'))






def main():

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='unlabeled2017',
        help='image dir')
    parser.add_argument('--style_dir', type=str, default='style', 
        help='style dir')
    parser.add_argument('--output_dir', type=str, default='output', 
        help='output dir')
    parser.add_argument('--train_size', type=int, default=10000,
        help='dataset train size')
    parser.add_argument('--test_size', type=int, default=1000,
        help='dataset test size')
    parser.add_argument('--gpu', type=int, default=-1,
        help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=16,
        help='batch size')
    parser.add_argument('--epoch', type=int, default=5,
        help='epoch num')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
        help='learning_rate')
    parser.add_argument('-t', type=int, default=2,
        help='t enumerate one origin')
    parser.add_argument('--save_epoch', type=int, default=10,
        help='save epoch per time')
    parser.add_argument('--eval_epoch', type=int, default=5,
        help='save epoch per time')
    parser.add_argument('--alpha', type=float, default=1,
        help='save epoch per time')
    parser.add_argument('--beta', type=float, default=1,
        help='save epoch per time')
    parser.add_argument('--gamma', type=float, default=1e-9,
        help='save epoch per time')
    parser.add_argument('--from_checkpoint', type=str, default=None,
        help='load with checkpoint')
    parser.add_argument('--from_check_no_optimizer', type=str, default=None,
        help='load with checkpoint without optimizer')
    parser.add_argument('--do_my_train_do_my_style', action='store_true')
    parser.add_argument('--train_style_image', type=str, default=None)
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--predict_dir', type=str, default=None)

    args = parser.parse_args()

    # set GPU
    if torch.cuda.is_available() and args.gpu >= 0:
        args.device = torch.device(f'cuda:{args.gpu}')
        print(f'[>] Using CUDA {args.gpu}')
    else:
        args.device = 'cpu'
        print('[>] Using cpu')

    # predict
    if args.do_predict:
        checkpoint = torch.load(args.from_checkpoint, map_location='cpu')
        model = checkpoint['model']
        optimizer_auto = checkpoint['optimizer_auto']
        optimizer_style = checkpoint['optimizer_style']
        style_images = StyleSet(args.style_dir)
        train_images = StyleSet(args.predict_dir)
        trainer = Trainer(args, model, style_images, optimizer_auto, optimizer_style, None)
        trainer.predict(train_images)
        exit(0)


    train_dataset = DataSet(image_dir=args.image_dir,
            num=args.train_size)
    test_dataset = DataSet(image_dir=args.image_dir,
        num=args.test_size)

    style_images = StyleSet(args.style_dir)

    if args.from_checkpoint != None:
        checkpoint = torch.load(args.from_checkpoint, map_location='cpu')
        model = checkpoint['model']
        optimizer_auto = checkpoint['optimizer_auto']
        optimizer_style = checkpoint['optimizer_style']
    elif args.from_check_no_optimizer != None:
        checkpoint = torch.load(args.from_check_no_optimizer)
        model = checkpoint['model']
        optimizer_auto = AdamW(model.parameters(), lr=args.learning_rate)
        optimizer_style = AdamW(model.parameters(), lr=args.learning_rate)
    else:
        model = Paper_StyleBank(total_style=len(style_images))
        optimizer_auto = AdamW(model.parameters(), lr=args.learning_rate)
        optimizer_style = AdamW(model.parameters(), lr=args.learning_rate)
    loss = MyLoss(args.alpha,args.beta,args.gamma)
    trainer = Trainer(args, model, style_images, optimizer_auto, optimizer_style, loss)
    print(f'[>] start Trainning🚂🚂🚂🚂')
    if args.do_my_train_do_my_style:
        trainer.trian_new_style(train_dataset, test_dataset, args.train_style_image)
    else:
        trainer.fit(train_dataset, test_dataset)
    


if __name__ == '__main__':
    main()
