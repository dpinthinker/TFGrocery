toco跟tensorflow是一起下载的，笔者通过brew安装的python，pip安装tf-nightly后toco路径如下：

```c
/usr/local/opt/python/Frameworks/Python.framework/Versions/3.6/bin
```

实际上，应该是/usr/local/Cellar/python/3.6.5_1/Frameworks/Python.framework/Versions/3.6/bin，但是软链接到了如上路径。

如果命令行不能执行到toco，则添加到环境变量，在~/.bash_profile添加如下行：

```c
 export PATH="/usr/local/opt/python/Frameworks/Python.framework/Versions/3.6/bin:$PATH"                                                                       
```
