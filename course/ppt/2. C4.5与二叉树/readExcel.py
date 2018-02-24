import pandas as pd
from pandas import read_excel
import numpy as np

read_excel
dataFrame=read_excel("贷款数据.xlsx",index_col=0)
print(dataFrame)
print(dataFrame.values)
