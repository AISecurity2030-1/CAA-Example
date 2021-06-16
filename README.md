# CAA-复合对抗攻击

## 1. 运行环境(python)

* torch >= 1.3.0
* torchvision
* advertorch
* tqdm
* pillow
* imagenet

## 2. 运行说明

### 2.1 Linf Attack

收集该攻击所需的[训练模型](https://www.dropbox.com/s/c9qlt1lbdnu9tlo/cifar_linf_8.pt?dl=0)并将其放在`checkpoints`文件下，然后用以下命令运行(注意，如果报gpu内存不足的错误，要修改默认的batchsize大小)：

```
python test_attacker.py --batch_size 512 --dataset cifar10 --net_type madry_adv_resnet50 --norm linf
```

### 2.2 L2 Attack

收集该攻击所需的 [训练模型](https://www.dropbox.com/s/1zazwjfzee7c8i4/cifar_l2_0_5.pt?dl=0)并将其放在`checkpoints`文件下，然后用以下命令运行(注意，如果报gpu内存不足的错误，要修改默认的batchsize大小)：

```
python test_attacker.py --batch_size 384 --dataset cifar10 --net_type madry_adv_resnet50_l2 --norm l2 --max_epsilon 0.5
```

### 2.3 Unrestricted Attack

该代码仅用于在bird_or_bicycle 数据集上进行测试，您也可以对其进行调整以适应您自己的数据集和任务。对于那些想要在bird_or_bicycle 数据集上运行的人，您必须首先安装`bird_or_bicycle`：

```
git clone https://github.com/google/unrestricted-adversarial-examples
pip install -e unrestricted-adversarial-examples/bird-or-bicycle

bird-or-bicycle-download
```

然后在[Unrestricted Adversarial Examples Challenge ](https://github.com/openphilanthropy/unrestricted-adversarial-examples)中收集不受限制的防御模型，例如[TRADESv2](https://github.com/xincoder/google_attack)，并将其放置到`checkpoints`. 最后，运行:

```
python test_attacker.py --batch_size 12 --dataset bird_or_bicycle --net_type ResNet50Pre --norm unrestricted
```

### 2.4 自定义攻击策略

您可以通过提供攻击字典列表来定义任意攻击策略：

例如，如果要组合`MultiTargetedAttack`,`MultiTargetedAttack`和`CWLinf_Attack_adaptive_stepsize`，只需定义如下列表：

```
[{'attacker': 'MultiTargetedAttack', 'magnitude': 8/255, 'step': 50}, {'attacker': 'MultiTargetedAttack', 'magnitude': 8/255, 'step': 25}, {'attacker': 'CWLinf_Attack_adaptive_stepsize', 'magnitude': 8/255, 'step': 125}]
```

CAA 还支持没有任何组合的单一攻击者：当您给出类似 的列表时`[{'attacker': 'ODI_Step_stepsize', 'magnitude': 8/255, 'step': 150}]`，这意味着您正在运行单一`ODI-PGD`攻击。

现在代码支持的攻击包括如下：

- GradientSignAttack
- PGD_Attack_adaptive_stepsize
- MI_Attack_adaptive_stepsize
- CWLinf_Attack_adaptive_stepsize
- MultiTargetedAttack
- ODI_Cos_stepsize
- ODI_Cyclical_stepsize
- ODI_Step_stepsize
- CWL2Attack
- DDNL2Attack
- GaussianBlurAttack
- GaussianNoiseAttack
- ContrastAttack
- SaturateAttack
- ElasticTransformAttack
- JpegCompressionAttack
- ShotNoiseAttack
- ImpulseNoiseAttack
- DefocusBlurAttack
- GlassBlurAttack
- MotionBlurAttack
- ZoomBlurAttack
- FogAttack
- BrightnessAttack
- PixelateAttack
- SpeckleNoiseAttack
- SpatterAttack
- SPSAAttack
- SpatialAttack

## 3. 注意点

* 建议环境在Linux下搭建，windows下有些地方会出错，比如那个bird_or_bicycle数据集在windows下使用会报错
* 程序中数据集的下载和加载位置根据您自己的情况去修改
* 程序中可能会用到的攻击训练模型原论文作者并没有都提供，如需要可以自行下载

## 4. 参考

* AAAI2021 论文[Composite Adversarial Attacks](https://arxiv.org/abs/2012.05434)
* https://github.com/vtddggg/CAA