# AI-Innovation-Practice
人工智能创新实践课程期中作业

### 如何运行？
运行main.py即可。
运行之前把--data_path改为数据集存放的路径
或者用命令行参数指定：
```python
python main.py --data_path /your/dataset/path/
```

### 如何添加模型？
在/model/中新建your_modelname.py，把继承nn.Module类的模型类放在里面就行了。

然后再model_interface.py里面import一下模型类。

再把 self.model = Model4Classifier(**kwargs) 替换成你的模型就行。

kwargs参数统一在main.py中声明和赋值。
```python
parser.add_argument("--data_path", type=str, default="../autodl-tmp/Real World/", help="Path to image dataset")
parser.add_argument("--image_size", type=int, default=224, help="Size of the input image to the model")
parser.add_argument("--image_channels", type=int, default=3, help="Number of channels in the input image")
parser.add_argument("--class_num", type=int, default=65, help="Dimensionality of the latent space")
parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone model to use')
parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of the model')
parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loaders")
parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the optimizer")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer")
parser.add_argument("--momentum", type=float, default=0.99, help="Momentum for the optimizer")
parser.add_argument("--lr_scheduler", type=bool, default=False, help="Use learning rate scheduler")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--model_name", type=str, default="ResNet", help="Name of the model to train")
parser.add_argument("--mode", type=str, default="train", help="Mode to run the script in: train or predict")
parser.add_argument("--k_fold", type=int, default=0, help="Number of folds for k-fold cross-validation")
parser.add_argument("--aug_type", type=str, default="default", help="Type of augmentation to use: default or light or strong")
```