import pandas as pd

train = pd.read_csv('train.csv', encoding = "latin1")
test = pd.read_csv('test.csv', encoding = "latin1")


train_1 = train[:25000]
test_1 = train[25000:]
train_1.to_csv("train_1.csv")
test_1.to_csv("test_1.csv")



ID = train_1['id'].Field(sequential=False,
                         use_vocab=False)

TEXT = train_1['anchor'].Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True
                  )

LABEL = train_1['anchor'].Field(sequential=False,
                   use_vocab=False,
                   is_target=True
                   )
train_2 = train_1.split(path = "./train_2.csv",format = "csv",
                  fields=[('id', ID),
                          ('text', TEXT),
                          ('label', LABEL)], skip_header=True)