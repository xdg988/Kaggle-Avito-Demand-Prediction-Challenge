import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from sklearn.cross_validation import KFold
import time
from sklearn.linear_model import Ridge

NFOLDS = 5
SEED = 42


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool = True):
        if(seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
        
def get_oof(clf, x_train, y, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        print('\nFold {}'.format(i))
        x_tr = x_train.iloc[train_index,:]
        y_tr = y[train_index]
        x_te = x_train.iloc[test_index,:]
        y_val = y[test_index]

        clf.train(x_tr, y_tr)

        val_pred=clf.predict(x_te)
        oof_train[test_index] += val_pred
        oof_test_skf[i, :] = clf.predict(x_test)
        
        cv_rms = sqrt(mean_squared_error(y_val.values, val_pred))
        print('fold cv {} RMSE score is {:.6f}'.format(i, cv_rms))

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

print('lv1...')
gbm=pd.read_csv('../input/stack1-lightgbm/gbm1.csv')
nn=pd.read_csv('../input/stack1-nn/nn.csv')
ranf=pd.read_csv('../input/stack1-random-forest/rf2.csv')
xg=pd.read_csv('../input/stack1-xgboost/xg2.csv')
rid=pd.read_csv('../input/stack1-ridge/ridge2.csv')
lv1=pd.merge(nn,gbm,on='item_id',how='left')
lv1=lv1.merge(ranf,on='item_id',how='left')
lv1=lv1.merge(xg,on='item_id',how='left')
lv1=lv1.merge(rid,on='item_id',how='left')
idx=lv1['item_id']

lv1_test=pd.read_csv('../input/avito-demand-prediction/sample_submission.csv')
lv1_test=lv1_test.merge(lv1,on='item_id',how='left')
ntest=lv1_test.shape[0]
lv1_test.drop(['item_id', 'deal_probability'],axis=1,inplace=True)

lv1_train=pd.read_csv('../input/avito-demand-prediction/train.csv',usecols=['item_id','deal_probability'])
ntrain=lv1_train.shape[0]
lv1_train=lv1_train.merge(lv1,on='item_id',how='left')
y_train=lv1_train.deal_probability.copy()
lv1_train.drop(['item_id', 'deal_probability'],axis=1,inplace=True)

kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)
ridge_params = {'alpha':25, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}
ridge= SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
ridge_oof_train, ridge_oof_test = get_oof(ridge, lv1_train, y_train, lv1_test)

rms = sqrt(mean_squared_error(y_train, ridge_oof_train))
print('Ridge OOF RMSE: {}'.format(rms))

print("Modeling Stage")

ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])

sub = pd.DataFrame(ridge_preds,columns=['ridge_pred'])
sub1 = pd.DataFrame(idx,columns=['item_id'])
sub1=sub1.set_index(sub.index)
ridge1=pd.concat([sub1,sub],axis=1)
ridge1.to_csv('ridge22.csv', index=False)

subm=pd.read_csv('../input/avito-demand-prediction/sample_submission.csv')
subm['deal_probability']=np.clip(ridge_oof_test,0,1)
subm.to_csv('ridgetest22.csv',index=False)