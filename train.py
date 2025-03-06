import os
import math
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data.dataloader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from my_dataset import MyDataSet #导入数据集
from ViT_CPE import vit_base_patch16_224_in21k as create_model #导入模型
from utils import read_split_data, train_one_epoch, evaluate #导入工具

#参数设置
classes = 3 #类别数
epochs = 1 #训练轮次
batch_size = 8 #一轮训练多少数据
lr = 0.001 #学习率
lrf = 0.01 #学习率的余弦衰减系数
path = "./" #数据集路径

print(f"epoch: {epochs}")

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 创建权重文件夹
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    # 初始化TensorBoard
    tb_writer = SummaryWriter()
    # 读取数据集的路径和标签
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

# 增加数据预处理方法

    # 定义训练集和验证集的图像预处理函数

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # 其他增强方法可以通过自定义的transforms加入
            transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.1),  # 颜色变换
            transforms.RandomRotation(30),  # 随机旋转
            transforms.RandomErasing(p=0.25, scale=(0.02, 1.0 / 3), ratio=(0.3, 3.3)),  # 随机擦除
            transforms.ToTensor(),
            transforms.Normalize([123.675/255, 116.28/255, 103.53/255], [58.395/255, 57.12/255, 57.375/255])
        ]),
        
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([123.675/255, 116.28/255, 103.53/255], [58.395/255, 57.12/255, 57.375/255])
        ])
    }

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                            images_class=train_images_label,
                            transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    # 设置数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=nw,
                                            collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=nw,
                                            collate_fn=val_dataset.collate_fn)

    # 创建模型
    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    if args.weights != "": #检查是否有预训练权重路径
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device) #加载模型预训练权重
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False)) # 将权重加载到模型

    if args.freeze_layers: #检查是否需要冻结权重
        for name, para in model.named_parameters(): #遍历模型的所有参数
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad] #获取需要梯度的参数
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5) #创建SGD优化器
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf) #创建学习率调度器


    best_val_loss = float('inf')  # 初始化最好的验证损失为正无穷

    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # 验证
        val_loss, val_acc = evaluate(model=model,
                                    data_loader=val_loader,
                                    device=device,
                                    epoch=epoch)



        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # 如果当前验证损失小于最好的验证损失，则保存模型权重
        if val_loss < best_val_loss:
            best_val_loss = val_loss  # 更新最好的验证损失
            torch.save(model.state_dict(), "./weights/best_model.pth")  # 保存最佳模型权重
        # 导出为ONNX格式
# 导出为ONNX格式
            dummy_input = torch.randn(1, 3, 224, 224).to(device)  # 将 dummy_input 移动到与模型相同的设备
            torch.onnx.export(model, dummy_input, "./weights/best_model.onnx",
                            export_params=True,
                            opset_version=12,  # 可以根据需要更改ONNX的版本
                            do_constant_folding=True,  # 是否做常量折叠
                            input_names=['input'],  # 输入名称
                            output_names=['output'],  # 输出名称
                            dynamic_axes={'input': {0: 'batch_size'},  # 动态batch_size
                                            'output': {0: 'batch_size'}})

if __name__ == '__main__':
    parser = argparse.ArgumentParser() #创建命令行参数解析器
    parser.add_argument('--num_classes', type=int, default = classes)
    parser.add_argument('--epochs', type=int, default = epochs)
    parser.add_argument('--batch-size', type=int, default = batch_size)
    parser.add_argument('--lr', type=float, default = lr)
    parser.add_argument('--lrf', type=float, default= lrf)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default= path)
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default= weights,
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
