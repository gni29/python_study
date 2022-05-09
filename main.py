import pandas as pd

train = pd.read_csv('train.csv', encoding = "latin1")
test = pd.read_csv('test.csv', encoding = "latin1")


train_1 = train[:25000]
test_1 = train[25000:]
train_1.to_csv("train_1.csv")
test_1.to_csv("test_1.csv")

from torchtext.legacy import data
from konply.tag import Mecab
tokenizer = Mecab()

ID = data.Field(sequential=False,
                use_vocab=False)

TEXT = data.field(sequential=True,
                  use_vocab=True,
                  tokenize=tokenizer.morphs,
                  lower=True
                  )

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   is_target=True
                   )
train_data = data.TabularDataset(train = "Train",format='csv',
                                    fields=[('id', ID),
                                            ('text',TEXT),
                                            ('label', LABEL)],
                                    skip_header = True)
