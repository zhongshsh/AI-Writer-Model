# AI Writer


使用 AI 实现文章续写。基于魔改的 GPT 模型，在不牺牲效果的前提下，将传统 GPT 模型的硬件门槛降低了1000倍，实现写作机器人的定制。 本项目基于 oneflow，集成了数据集、模型训练、模型部署，能通过已训练好的模型快速体验 AI 续写效果，也能通过微调模型、更改数据集等方式进行模型定制。 


## 1. 效果示意
![python_run_py](https://oneflow-static.oss-cn-beijing.aliyuncs.com/ai_writer/python_run_py.png)



## 2. 目录结构

```
.
|-- README.md                      # 项目说明文件
|-- src                            # 模型文件夹
|-- data                           # 数据文件夹
|-- log                            # 日志文件夹
|-- model                          # 模型存储文件夹
|-- pictures                       # 相关图片
|-- init.sh                        # 环境初始化文件
|-- config.py                      # 配置文件
`-- main.py                        # 训练和测试的主文件
```



## 3. 使用说明

项目支持模型定制和模型效果展示。


### 3.1 定制模型

如果想要训练一个自己期望的模型，请进行如下操作：

- 使用 `sh init.sh` 实现环境初始化
- 进入 data 目录，参考 t 数据集的格式来创建自己的数据集文件夹 my_data ，并将数据集文件命名为 train.txt 放置到 data 目录下的 my_data 文件夹中
- 修改 config 文件中的参数 `data_name` 的值为 `my_data` ，以及调整模型的其他参数
- 运行 train.py 文件，训练好的模型存储在 `./model/my_data/` 下


### 3.2 效果展示

- 修改 config 文件中的参数 `TRAINED_MODEL`，表示展示哪一个模型
- 运行 `main.py` 中的 test 即可以看到模型预测结果



> 模型原理参考：https://github.com/BlinkDL/RWKV-LM
