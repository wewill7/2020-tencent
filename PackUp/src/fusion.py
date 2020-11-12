import pandas as pd
import numpy as np

oof0=pd.read_csv('five_gender0.csv',index_col=0).values
oof1=pd.read_csv('five_gender1.csv',index_col=0).values
oof2=pd.read_csv('five_gender2.csv',index_col=0).values
oof3=pd.read_csv('five_gender3.csv',index_col=0).values
oof4=pd.read_csv('five_gender4.csv',index_col=0).values

oof10=pd.read_csv('five_age0.csv',index_col=0).values
oof11=pd.read_csv('five_age1.csv',index_col=0).values
oof12=pd.read_csv('five_age2.csv',index_col=0).values
oof13=pd.read_csv('five_age3.csv',index_col=0).values
oof14=pd.read_csv('five_age4.csv',index_col=0).values

oof_gender = (oof0+oof1+oof2+oof3+oof4)/5
oof_test_gender  = np.argmax(oof, axis=1)+1
oof_age = (oof10+oof11+oof12+oof13+oof14)/5
oof_test_age  = np.argmax(oof, axis=1)+1

submission = pd.read_csv('./submission.csv', index_col = 0 )
submission.predicted_gender = oof_test_gender
submission.predicted_age = oof_test_age
submission.to_csv('./submission.csv')