from keras.utils.generic_utils import Progbar

progbar = Progbar(100000)
for i in range(100000):                
	for j in range(10000):
		a=0
	progbar.add(1, values=[('aaa',0.3),('bbb',100)])


# model.fit(...) 函数出现的进度条效果一样
