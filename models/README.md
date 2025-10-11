# 模型目录

此目录用于存储训练好的模型。由于模型文件较大，它们未包含在git版本控制中。

## 模型文件

项目中使用的模型包括：

- `arxiv_gcn2_model.pth` - 用于arXiv数据集的GCN2模型
- `BAHouse_gcn_model.pth` - 用于BAHouse数据集的GCN模型
- `Cora_gcn2_model.pth` - 用于Cora数据集的GCN2模型
- `FacebookPagePage_gcn2_model.pth` - 用于Facebook页面-页面数据集的GCN2模型
- `PubMed_gcn2_model.pth` - 用于PubMed数据集的GCN2模型

## 如何获取模型

您可以通过以下方式获取预训练模型：

1. 运行训练脚本生成自己的模型：`./train.sh`
2. 联系项目维护者获取预训练模型
