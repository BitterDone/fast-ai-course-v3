#!/usr/bin/env python
# coding: utf-8

# ## MNIST CNN

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *


# ### Data

# In[ ]:


path = untar_data(URLs.MNIST)


# In[ ]:


path.ls()


# In[ ]:


il = ImageList.from_folder(path, convert_mode='L')


# In[ ]:


il.items[0]


# In[ ]:


defaults.cmap='binary'


# In[ ]:


il


# In[ ]:


il[0].show()


# In[ ]:


sd = il.split_by_folder(train='training', valid='testing')


# In[ ]:


sd


# In[ ]:


(path/'training').ls()


# In[ ]:


ll = sd.label_from_folder()


# In[ ]:


ll


# In[ ]:


x,y = ll.train[0]


# In[ ]:


x.show()
print(y,x.shape)


# In[ ]:


tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])


# In[ ]:


ll = ll.transform(tfms)


# In[ ]:


bs = 128


# In[ ]:


# not using imagenet_stats because not using pretrained model
data = ll.databunch(bs=bs).normalize()


# In[ ]:


x,y = data.train_ds[0]


# In[ ]:


x.show()
print(y)


# In[ ]:


def _plot(i,j,ax): data.train_ds[0][0].show(ax, cmap='gray')
plot_multi(_plot, 3, 3, figsize=(8,8))


# In[ ]:


xb,yb = data.one_batch()
xb.shape,yb.shape


# In[ ]:


data.show_batch(rows=3, figsize=(5,5))


# ### Basic CNN with batchnorm

# In[ ]:


def conv(ni,nf): return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)


# In[ ]:


model = nn.Sequential(
    conv(1, 8), # 14
    nn.BatchNorm2d(8),
    nn.ReLU(),
    conv(8, 16), # 7
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 32), # 4
    nn.BatchNorm2d(32),
    nn.ReLU(),
    conv(32, 16), # 2
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 10), # 1
    nn.BatchNorm2d(10),
    Flatten()     # remove (1,1) grid
)


# In[ ]:


learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)


# In[ ]:


print(learn.summary())


# In[ ]:


xb = xb.cuda()


# In[ ]:


model(xb).shape


# In[ ]:


learn.lr_find(end_lr=100)


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(3, max_lr=0.1)


# ### Refactor

# In[ ]:


def conv2(ni,nf): return conv_layer(ni,nf,stride=2)


# In[ ]:


model = nn.Sequential(
    conv2(1, 8),   # 14
    conv2(8, 16),  # 7
    conv2(16, 32), # 4
    conv2(32, 16), # 2
    conv2(16, 10), # 1
    Flatten()      # remove (1,1) grid
)


# In[ ]:


learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)


# In[ ]:


learn.fit_one_cycle(10, max_lr=0.1)


# ### Resnet-ish

# In[ ]:


class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = conv_layer(nf,nf)
        self.conv2 = conv_layer(nf,nf)
        
    def forward(self, x): return x + self.conv2(self.conv1(x))


# In[ ]:


help(res_block)


# In[ ]:


model = nn.Sequential(
    conv2(1, 8),
    res_block(8),
    conv2(8, 16),
    res_block(16),
    conv2(16, 32),
    res_block(32),
    conv2(32, 16),
    res_block(16),
    conv2(16, 10),
    Flatten()
)


# In[ ]:


def conv_and_res(ni,nf): return nn.Sequential(conv2(ni, nf), res_block(nf))


# In[ ]:


model = nn.Sequential(
    conv_and_res(1, 8),
    conv_and_res(8, 16),
    conv_and_res(16, 32),
    conv_and_res(32, 16),
    conv2(16, 10),
    Flatten()
)


# In[ ]:


learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)


# In[ ]:


learn.lr_find(end_lr=100)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(12, max_lr=0.05)


# In[ ]:


print(learn.summary())


# ## fin

# In[ ]:




