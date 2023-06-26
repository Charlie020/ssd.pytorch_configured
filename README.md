# ssd.pytorch_configured

## 一、代码说明



### （1）本仓库为本人根据自己项目需要所配置好的pytorch版本的ssd模型。建仓库的初衷是本人在跑的过程当中遇到了很多的报错，因此想一起分享交流经验。本仓库修改了部分可能出现bug的代码，同时附上了解决部分问题的博客和交流贴



### （2）本仓库所修改的源代码的作者及其仓库链接：https://github.com/amdegroot/ssd.pytorch

<br/>

## 二、从零开始配置模型以及可能出现的问题的解决方案



以下步骤不需要都完成，根据自己的需要进行阅读



### 1、训练



（1）配置ssd：https://blog.csdn.net/m0_47452894/article/details/112783858 中的`运行eval.py`之前的部分，在配置好这一条的基础上运行`train.py`，看能不能成功运行。
  <br/>

（2）完善ssd：https://blog.csdn.net/dear_queen/article/details/114301614   （用于解决目标计算机积极拒绝的问题）
  <br/>

（3）解决问题`“Expected a 'cuda' device type for generator but found 'cpu'”`： https://github.com/amdegroot/ssd.pytorch/issues/561
  <br/>
  
（4）若你的数据集按VOC格式划分好了的，可跳过这一步。若未划分，则可以找到`data/VOCdevkit/VOC2007`下的`make.py`，先在代码中设定数据集各部分划分的比例（默认训练：验证：测试=6：2：2），然后运行。代码在`ImageSets/Main`下自动生成`trainval.txt`,`test.txt`,`train.txt`,`val.txt`四个文件，方便模型读取哪些图片用来训练，哪些用来验证，哪些用来测试。

`data`文件夹中的VOC数据集文件结构如下（Annotations中放数据集中所有的标签，ImageSets中放数据集中所有的图片）：
```
--data
|   --VOCdevkit
|   	  --VOC2007
|        	--Annotations
|             		--1.xml,2.xml,....
|        	--JPEGImages
|             		--1.jpg,2.jpg,....
|        	--ImageSets
|             		--Main
|                 	    --test.txt        # 用于测试，这几个文件中存的都是文件的索引，如1,2,3,4...加上.xml或者.jpg就是对应标签和图像的名字
|                 	    --train.txt       # 用于训练
|                 	    --trainval.txt    # 训练加验证
|                 	    --val.txt         # 验证
|        	--debug.py
|        	--make.py
|               --clear.py
|   ······ # 其他文件
```
<br/>
（5）解决问题`“IndexError: too many indices for array”`： https://github.com/amdegroot/ssd.pytorch/issues/224  （此为数据集当中可能出现的问题，交流帖中代码已保存至仓库中`data/VOCdevkit/VOC2007/debug.py`，即与VOC2007数据集中的`Annotations`、`JPEGImages`等在同一目录，运行后在哪一个xml停下来，就去删除或修改对应的xml和图片，当运行`debug.py`不再显示` “ INDEX ERROR HERE ! ”`，再运行`train.py`则不会报该项错误） <br/><br/>

（6）clear.py用来解决图片与标签数量不匹配的问题。
<br/><br/><br/>

一般而言，完成上述部分即可开始训练了。
<br/>
<br/>
<br/>
### 2、测试

先按照上述（1）中的`运行eval.py`修改，运行eval.py，若没有成功运行则可能会报`TypeError: forward() takes 4 positional arguments but 9 were given`的错，因此若遇到该项报错，只需按照以下内容修改即可。


（1）修改`eval.py`

```python
#修改39行；将源码的模型文件改成自己训练好的模型文件
parser.add_argument('--trained_model',
                    default='weights/ssd300_VOC_20000.pth', type=str, 
                    help='Trained state_dict file path to open')

#将71行的
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets','Main', '{:s}.txt')
#修改为
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets', 'Main') + os.sep + '{:s}.txt'
#如不修改，则imgsetpath 可能无法正常拼接出来，导致报错

```



（2）按照以下内容修改`ssd.py`：

```python
# 51行附近
	if phase == 'test':
     	self.softmax = nn.Softmax()
     	self.detect = Detect()
        
        
# 120行附近        
    if self.phase == "test":
            output = self.detect.forward(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
```


完成上述部分，看能否运行成功`eval.py`，若报错，下面的部分可能会有所帮助。


（3）可能遇到`”RuntimeError: index_select(): functions with out=... arguments don't support automatic differentiation, but one of the arguments requires grad.”`，解决问题：https://blog.csdn.net/XiaoGShou/article/details/125253471
<br/>


（4）可能遇到
```
cv2.error: OpenCV(4.5.2)  :-1 error: (-5:Bad argument) in function 'rectangle`
\> Overload resolution failed:
\> - Can't parse 'pt1'. Sequence item with index 0 has a wrong type
\> - Can't parse 'pt1'. Sequence item with index 0 has a wrong type
······
```

解决问题：https://stackoverflow.com/questions/67921192/5bad-argument-in-function-rectangle-cant-parse-pt1-sequence-item-wit
<br/>


（5）可能遇到`”AttributeError: ‘NoneType‘ object has no attribute ‘text‘“`，解决问题：https://blog.csdn.net/qq_55535816/article/details/121456901 ，将代码报错部分按照此博客中方法二的方法设置

（6）可能遇到` R = [obj for obj in recs[imagename] if obj['name'] == classname] KeyError: '1000'`，解决问题：https://github.com/amdegroot/ssd.pytorch/issues/482 （删除`annotations_cache/annots.pkl`）

<br/><br/>
### 3、注：测试时修改后的ssd.py的代码并没法在训练的时候也成功运行，因此以后想要反复训练和测试，仅需在相应阶段使用相应的代码，这里已经整理好了，仅需调整`ssd.py`的内容（下面代码为训练时用，测试部分已被注释）：

（训练时，注释掉测试用的代码，同时取消注释训练用的代码；测试时同理）

```
# 51行附近
	# 训练时代码
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
                    # ORIGINAL IMPLEMENTATION DEPRECATED
                    # self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
            self.detect = Detect()  # corrected implementation
                    # my comments
                    # this 'Detect' Function is not compatible with new Pytorch version,
                    # generates error 'Legacy autograd function with non-static forward
                    # method is deprecated. Please use new-style autograd function with
                    # static forward method.'
                    # Correction is implemented by passing above arguments directly to
                    # command .apply() at the forward method below.
        """
        # 测试时代码
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.25, 0.45)
        """


# 120行附近
	# 训练时代码
        if self.phase == "test":
            # ORIGINAL LINE IS DEPRECATED
            # output = self.detect(
            # corrected implementation:
            output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
               # loc preds
               loc.view(loc.size(0), -1, 4),
               # conf preds
               self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
               # default boxes
               self.priors.type(type(x.data))
           )
        # 测试时代码
            """
        if self.phase == "test":
            output = self.detect.forward(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
            """ 
```
