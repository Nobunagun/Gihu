#一、运行说明   
　　1.本OCR工程是在windows平台下使用Python-3.6和tensorflow-1.2.1完成的。    
　　2.本工程文件中的src中的py文件均放在根目录下，训练好的模型文件在**model**中。    
　　3.运行时，调用方式也更改为**./run.sh**，请在根目录下**test**文件夹中放入需要进行预测的对象图片，最后输出的csv文件会保存至**output**文件夹中。         
　　4.如果run.sh运行失败，也可以直接运行根目录下的**predicate.py**文件*。       
　　5.如果程序无法获取根目录的路径，请将config.py中第34行的变量**IMAGE_DIR **赋值为**根目录的路径**   
#二、模型说明    
>OCR一般包含两步: 1. detection-->找到包含文字的区域(proposal); 2. classification-->识别区域中的文字。
		考虑本题目中的文字区域已实现标记好，因此无需进行CTPN这一步，直接对图片进行识别。这一方面的工作近年来采用cnn+rnn的模型比较火热，最后结合CTC_loss进行非定长的loss计算。
		CNN中常采用resnet系列(densenet)和inception系列，RNN中常用双向LSTM，对于较长的识别任务CTC_loss常会给出较好的结果，seq2seq_loss结合attention机制在较短文字的识别也会给出不错的结果。
		且Github上已经有较多的类似训练模型。考虑到两点1.densenet需要计算资源较多以及时间成本也较高；2.我们想自己搭建一套模型，不想使用别人的Fine-tuning模型，因此我们选择了白翔老师在2015年给出的一个轻量级的时间序列网络CRNN，具体结构如下：    
		
<div align=center>![image](https://github.com/Nobunagun/Gihu/blob/master/1410963932c7303517345976372c5d56_681e5ffa-d31d-402b-ac6d-4aed845817b9.jpg?raw=true)     	
<div align=center>![enter image description here](https://github.com/Nobunagun/Gihu/blob/master/285567223310761237.png?raw=true)     	
		
　　本工程所采用的模型为CNN+biLSTM+CTC Loss来完成这两个步骤，从而实现端到端的文字识别，网络模型如图所示：   
　　　

　　模型的定义文件在根目录下的**model_crnn.py**文件中。网络输入的张量形状为(batch_size, 512, 32, 1)，，CNN将输入图片在长度和宽度方向上进行特征提取，其中Max pooling中的窗口大小为1x2，保证提出的特征具有横向的长度，有利于比较长的文本的识别，同时加入了BatchNorm，有助于模型收敛，CNN输出张量形状为(batch_size, 125, 1, 512)；之后添加了2层biLSTM，选用GRU单元，输出张量形状为(batch_size, 125, 512)，经过一个全连接层将输入从lstm隐层空间映射至标签空间，其输出张量形状为(125,batch_size,VOCNUM)，其中VOCNUM为标签字典的长度，最后计算出CTC Loss（输出字符串最大64个长度），通过Adam方法训练网络。    
		
　　事实上，我们还采用了两种改进方式：1.CNN采用densenet，2.两层Bilstm均采用res方式连接；但是由于后来资源和时间有限，这两个模型均没有训练完毕，前者只到80%左右，后者约87%。    
#三、训练的一些说明    

##1. 训练数据预处理与增强   
　　将初赛和复赛的训练样本作为一单位的训练数据，将样本图片通过灰度变化和尺寸调整（调整为32x512的统一大小）进行预处理；同时为了增加训练样本的数量，又采用了图片缩放、旋转、裁剪、模糊、仿射变换以及添加椒盐噪声等方法（**data_gen.py中的random_image函数**）生成了新的样本（**config.py中的train_data_augmentation函数**，提交的模型的训练中随机化了7个单位的训练数据，见**config.py中的RANDOM_SIZE = 7**）。    　
##2. 训练过程   
　　训练过程中，我们采用了CNN和RNN分别置一定大小的学习率（见train_crnn中），CNN部分稍大，RNN部分稍小，可以取得更高的训练效率。随着后期RNN继续降低学习率（从1e-7降至1e-8）准确率又会稍微提升一些，（约从92%-92.5%）。   
##3. 预测数据处理   
　　对于过长的图片，在通过训练好的模型进行预测的时候，往往只能成功预测前半段文字。为了解决这个问题，于是对长宽比超过一定值的图片进行了分割，根据分割比例在图片长度方向搜索灰度值最大（即看上去越白）的一列作为分割线（**cofig.py中的predicate_data函数**），从而实现将过长的图片分割成了2部分进行预测，能有效提高预测精度（可以在原基础上92.5%左右再提升至（93.6%）约1%的效果）。   
