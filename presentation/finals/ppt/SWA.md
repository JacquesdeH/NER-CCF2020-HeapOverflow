**SWA**

1. 提出的论文： [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407): Pavel Izmailov et. al 
2.  优化算法，如：SGD，ADAM
3.  主要过程使用了SGD算法。
4.  参数：weight $\hat{w}$，LR bounds $\alpha_1, \alpha_2$, cycle length $c$, number of iterations $n$
5. ![image-20201222214144254](C:\Users\8igfive\AppData\Roaming\Typora\typora-user-images\image-20201222214144254.png)

6. 首先用权重$\hat{w}$初始化一个权重$\hat{w}_{SWA}$，优化过程是对权重$\hat{w}$使用$n$次SGD算法进行梯度下降，每$c$次用当前的$\hat{w}$与$\hat{w}_{SWA}$做一次加权平均去更新$\hat{w}_{SWA}$。

7. 论文中指出，使用这个方法可以比传统的SGD算法收敛到更好的结果，以下是示例：

   ![image-20201222224726329](C:\Users\8igfive\AppData\Roaming\Typora\typora-user-images\image-20201222224726329.png)

   在使用固定学习率的SGD进行了140个epoch后，换成SWA便快速收敛到了一个相对来说好很多的结果。

   （蓝色线是使用变学习率的SGD算法，绿色线是固定学习率的SGD算法，红色线是固定学习率的SWA算法）





**需要在PPT中提到的内容：**

1. SWA算法是一个优化算法，输入的参数包括需要优化的权重$\hat{w}$，学习率的上下界$\alpha_1,\alpha_2$，周期长度$c$，迭代次数$n$。

2. 算法过程为：首先用权重$\hat{w}$初始化一个权重$\hat{w}_{SWA}$，优化过程是对权重$\hat{w}$使用$n$次SGD算法进行梯度下降，每$c$次用当前的$\hat{w}$与$\hat{w}_{SWA}$做一次加权平均去更新$\hat{w}_{SWA}$。

3. 算法的优势：论文中指出，使用这个方法可以比传统的SGD算法收敛到更好的结果，以下是示例：

   ![image-20201222224726329](C:\Users\8igfive\AppData\Roaming\Typora\typora-user-images\image-20201222224726329.png)

   在使用固定学习率的SGD进行了140个epoch后，换成SWA便快速收敛到了一个相对来说好很多的结果。

