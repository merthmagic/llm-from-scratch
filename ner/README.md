# NER

Named Entity Recognition

可能的实现方法：

- 预训练的语言模型 + FC

## BERT实现NER

使用BERT预训练模型实现命名实体识别（NER）的基本步骤如下：

步骤1：环境准备
确保你已经安装了Python和以下的库：

transformers
torch
seqeval（用于评估）
pandas（用于数据处理）
可以通过简单的pip命令安装这些库：

pip install transformers torch seqeval pandas
步骤2：数据准备
你需要有一个用于NER任务的已标注数据集。数据通常是以BIO（Beginning, Inside, Outside）标签格式标注的，例如：

句子: 哈利 波特 和 赫敏 是 好 友。
标签: B-PER I-PER O B-PER O O O
数据需要被转换成BERT能理解的格式，通常是两个列表，一个包含分词后的句子，另一个包含对应的标签。

步骤3：加载预训练的BERT模型
使用transformers库加载预训练的BERT模型和分词器：

from transformers import BertTokenizer, BertForTokenClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
num_labels是你的标签数量（包括O标签）。

步骤4：数据预处理
使用BERT分词器将文本和标签转换为模型可接受的格式。

inputs = tokenizer(texts, padding=True, truncation=True, is_split_into_words=True, return_tensors="pt")
labels = torch.tensor([label_ids])  # label_ids是将文本标签转换为ID的列表
步骤5：模型训练
定义优化器和训练循环，然后开始训练模型。

from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        inputs = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
步骤6：模型评估
在验证集或测试集上评估你的模型性能。

from seqeval.metrics import precision_score, recall_score, f1_score

评估代码示例
model.eval()
predictions, true_labels = [], []
for batch in val_dataloader:
    inputs = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']

    with torch.no_grad():
        outputs = model(inputs, attention_mask=attention_mask)
    
    logits = outputs.logits
    predictions.extend([list(p) for p in np.argmax(logits.detach().cpu().numpy(), axis=2)])
    true_labels.extend(labels.detach().cpu().numpy())

计算精确度、召回率和F1分数
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
步骤7：模型部署
将训练好的模型保存并在实际应用中使用。

model.save_pretrained('my_ner_model')
tokenizer.save_pretrained('my_ner_model')
这些是实现NER任务的高级步骤。实际上，每一步都有很多细节需要注意，例如对数据进行预处理时标签与分词后的词对齐问题、处理特殊字符、选择合适的学习率等。你可能还需要调整模型的超参数，比如批量大小、学习率和训练的轮数，以获得最佳性能。