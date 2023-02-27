# list = [1,2,3,4]
# for i in list:
#     print(i)
##但是实际上这种写法非常的反直觉...


d = {"a":1,"b":2}
for i in d:
    print(i)  ##实际上你通过字典迭代拿到的是key的元组  注意 3.8以后字典也是有序打印的了..垃圾python



with open("exp.txt") as f:
    for i in f:
        print(f)
##文件这么复杂的东西为啥也可以对其进行相应的迭代的操作


##iterable(这个是个名词！！！！这个是可以迭代的对象) for loop 就是要一个一个的从这里往外出去拿，这就是比如我们的list就是这种形式
#这个概念更像是一个容器，container  __iter__ __getitem__
##for loop 后面一定是一个iterable 但是实际上你是要经过一个iterator

# iterator(迭代器) 必须有__next__(指针)