import numpy as np
from scipy.optimize import minimize
from statsmodels.tsa.statespace.mlemodel import MLEModel

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.statespace.msm import MarkovRegression
from scipy.optimize import minimize
from scipy.stats import chi2
import pandas as pd

param = {}

def hamilton_filter_endo_switch(param, dd, dS):
    y = dd['Y']
    x1, x2, x3 = dd['X.0.m'], dd['X.1.m'], dd['X.2.m']
    x1S, x2S, x3S = dS['X.0.m'], dS['X.1.m'], dS['X.2.m']

    nobs = dd.shape[0]

    const0, const1 = param[0], param[1]

    # для рядов более высокой частоты
    beta1_0, beta1_1 = param[2], param[3]
    beta2_0, beta2_1 = param[4], param[5]
    beta3_0, beta3_1 = param[6], param[7]

    sigma0, sigma1 = param[8], param[9]

    # для рядов более низкой частоты
    a0_0, a1_0, a2_0, a3_0 = param[12:16]
    a0_1, a1_1, a2_1, a3_1 = param[16:20]

    # марковская цепь?
    p00 = 1 / (1 + np.exp(-(a0_0 + a1_0 * x1S[0] + a2_0 * x2S[0] + a3_0 * x3S[0])))
    p11 = 1 / (1 + np.exp(-(a0_1 + a1_1 * x1S[0] + a2_1 * x2S[0] + a3_1 * x3S[0])))
    p01, p10 = 1 - p00, 1 - p11

    ps_i0, us_j0 = None, None
    ps_i1, us_j1 = None, None

    sspr1 = (1 - p00) / (2 - p11 - p00)
    sspr0 = 1 - sspr1

    us_j0 = sspr0
    us_j1 = sspr1

    ps = np.zeros((nobs, 2))
    fp = np.zeros((nobs, 2))
    loglike = 0

    for t in range(1, nobs):
        ps_i0 = us_j0
        ps_i1 = us_j1

        p00 = 1 / (1 + np.exp(-(a0_0 + a1_0 * x1S[t - 1] + a2_0 * x2S[t - 1] + a3_0 * x3S[t - 1])))
        p11 = 1 / (1 + np.exp(-(a0_1 + a1_1 * x1S[t - 1] + a2_1 * x2S[t - 1] + a3_1 * x3S[t - 1])))
        p01, p10 = 1 - p00, 1 - p11

        # residuals
        er0 = y[t] - const0 - beta1_0 * x1[t] - beta2_0 * x2[t] - beta3_0 * x3[t]
        er1 = y[t] - const1 - beta1_1 * x1[t] - beta2_1 * x2[t] - beta3_1 * x3[t]

        eta_j0 = (1 / (np.sqrt(2 * np.pi * sigma0**2))) * np.exp(-(er0**2) / (2 * (sigma0**2)))
        eta_j1 = (1 / (np.sqrt(2 * np.pi * sigma1**2))) * np.exp(-(er1**2) / (2 * (sigma1**2)))

        f_yt = ps_i0 * p00 * eta_j0 + ps_i0 * p01 * eta_j1 + ps_i1 * p10 * eta_j0 + ps_i1 * p11 * eta_j1

        if f_yt < 0 or np.isnan(f_yt):
            loglike = -100000000
            break


        us_j0 = (ps_i0 * p00 * eta_j0 + ps_i1 * p10 * eta_j0) / f_yt
        us_j1 = (ps_i0 * p01 * eta_j1 + ps_i1 * p11 * eta_j1) / f_yt

        ps[t] = [ps_i0, ps_i1]
        fp[t] = [us_j0, us_j1]
        loglike += np.log(f_yt)

    return {'loglike': -loglike, 'fp': fp, 'ps': ps}


def tf2(param, dd, dS):
    return hamilton_filter_endo_switch(param, dd, dS)['loglike']


def MS_MIDAS_endo(dataModel, dataState, nLags=0):
    X = dataModel.iloc[:, 1].values
    Y = dataModel['Ydata'].dropna().values

    tmp = pd.concat([mls(X, k=0, m=3), mls(X, k=1, m=3), mls(X, k=2, m=3)], axis=1)
    tmp['Y'] = Y

    XSwitch = dataState.iloc[:, 1].values
    tmpSwitch = pd.concat([mls(XSwitch, k=0, m=3), mls(XSwitch, k=1, m=3), mls(XSwitch, k=2, m=3)], axis=1)
    tmpSwitch['Y'] = Y

    mLin = ARIMA(tmp['Y'], order=(0, 0, 0)).fit()
    modMSwM = MarkovRegression(tmp['Y'], k_regimes=2, exog=tmp.drop('Y', axis=1)).fit()
    
    param = np.concatenate([modMSwM.params, modMSwM.scale, 
                             [-np.log(1/modMSwM.transition[0, 0] - 1), 
                              -np.log(1/modMSwM.transition[1, 1] - 1), 
                              [-np.log(1/modMSwM.transition[0, 0] - 1), 0, 0, 0], 
                              [-np.log(1/modMSwM.transition[1, 1] - 1), 0, 0, 0]])

    def obj_func(param):
        return -hamilton_filter_endoSwitch(param, tmp, tmpSwitch)['loglike']
    
    iter0 = minimize(obj_func, x0=param, args=(tmp, tmpSwitch), method='BFGS')
    iter1 = minimize(obj_func, x0=iter0.x, args=(tmp, tmpSwitch), method='BFGS')
    iter2 = minimize(obj_func, x0=iter1.x, args=(tmp, tmpSwitch), method='BFGS')
    iter3 = minimize(obj_func, x0=iter2.x, args=(tmp, tmpSwitch), method='BFGS')
    iter3 = minimize(obj_func, x0=iter3.x, args=(tmp, tmpSwitch), method='BFGS')
    iter3 = minimize(obj_func, x0=iter3.x, args=(tmp, tmpSwitch), method='BFGS')
    
    resPar = iter3.x
    
    lUR = hamilton_filter_endoSwitch(resPar, tmp, tmpSwitch)['loglike']
    lR = hamilton_filter_endoSwitch(param, tmp, tmpSwitch)['loglike']
    
    LRTest_Endo_PV = 1 - chi2.cdf(2 * (lR - lUR), df=6)
    
    coef_Y_R1 = resPar[[0, 2, 4, 6]]
    coef_Y_R2 = resPar[[1, 3, 5, 7]]
    coef_Prob_00 = resPar[12:15]
    coef_Prob_11 = resPar[16:19]
    
    rr = {'coef_Y_R1': coef_Y_R1,
          'coef_Y_R2': coef_Y_R2,
          'coef_Prob_00': coef_Prob_00,
          'coef_Prob_11': coef_Prob_11,
          'LREndo': LRTest_Endo_PV,
          'endo_FP': hamilton_filter_endoSwitch(resPar, tmp, tmp)['fp'],
          'endo_PS': hamilton_filter_endoSwitch(resPar, tmp, tmp)['ps'],
          'exo_FP': hamilton_filter_endoSwitch(param, tmp, tmp)['fp'],
          'exo_PS': hamilton_filter_endoSwitch(param, tmp, tmp)['ps'],
          'dataModel': dataModel,
          'dataState': dataState,
          'resPar': resPar}

    class rrClass:
        pass

    rrClass.__dict__.update(rr)
    rr = rrClass

    pred_E = predict(rr)

    coef_Y_R1 = param[[0, 2, 4, 6]]
    coef_Y_R2 = param[[1, 3, 5, 7]]
    coef_Prob_00 = param[12:15]
    coef_Prob_11 = param[16:19]

    rr_exo = {'coef_Y_R1': coef_Y_R1,
              'coef_Y_R2': coef_Y_R2,
              'coef_Prob_00': coef_Prob_00,
              'coef_Prob_11': coef_Prob_11,
              'LREndo': LRTest_Endo_PV,
              'endo_FP': hamilton_filter_endoSwitch(param, tmp, tmp)['fp'],
              'endo_PS': hamilton_filter_endoSwitch(param, tmp, tmp)['ps'],
              'exo_FP': hamilton_filter_endoSwitch(param, tmp, tmp)['fp'],
              'exo_PS': hamilton_filter_endoSwitch(param, tmp, tmp)['ps'],
              'dataModel': dataModel,
              'dataState': dataState,
              'resPar': param}

    rr_exoClass = rrClass
    rr_exoClass.__dict__.update(rr_exo)
    rr_exo = rr_exoClass

    pred_exo = predict(rr_exo)

    MAE_endo = np.mean(np.abs(pred_E.yf_top - Y))
    MAE_exo = np.mean(np.abs(pred_exo.yf_top - Y))

    rr.MAE_endo = MAE_endo
    rr.MAE_exo = MAE_exo

    return rr


def predict_endoSwitchMIDAS(eMIDAS, newdataModel=None, newdataState=None):
    calcNew = True
    
    if newdataModel is None and newdataState is None:
        calcNew = False
    
    if not calcNew:
        newdataModel = eMIDAS.dataModel
        newdataState = eMIDAS.dataState
    
    lastObservedState = eMIDAS.endo_FP[-1, :] if not calcNew else eMIDAS.endo_FP[1, :]
    
    const0 = eMIDAS.resPar[0]
    const1 = eMIDAS.resPar[1]
    beta1_0 = eMIDAS.resPar[2]
    beta1_1 = eMIDAS.resPar[3]
    beta2_0 = eMIDAS.resPar[4]
    beta2_1 = eMIDAS.resPar[5]
    beta3_0 = eMIDAS.resPar[6]
    beta3_1 = eMIDAS.resPar[7]
    
    sigma0 = eMIDAS.resPar[8]
    sigma1 = eMIDAS.resPar[9]
    
    a0_0 = eMIDAS.resPar[12]
    a1_0 = eMIDAS.resPar[13]
    a2_0 = eMIDAS.resPar[14]
    a3_0 = eMIDAS.resPar[15]
    a0_1 = eMIDAS.resPar[16]
    a1_1 = eMIDAS.resPar[17]
    a2_1 = eMIDAS.resPar[18]
    a3_1 = eMIDAS.resPar[19]
    
    x1 = newdataModel.iloc[2::3, 1].values
    x2 = newdataModel.iloc[1::3, 1].values
    x3 = newdataModel.iloc[::3, 1].values
    
    x1S = newdataState.iloc[2::3, 1].values
    x2S = newdataState.iloc[1::3, 1].values
    x3S = newdataState.iloc[::3, 1].values
    
    nobs = len(x1)
    
    yf_w = np.zeros(nobs)
    yf_top = np.zeros(nobs)
    yf_r1 = np.zeros(nobs)
    yf_r2 = np.zeros(nobs)
    
    for t in range(nobs):
        # Transition probabilities at period t
        p00 = 1 / (1 + np.exp(-(a0_0 + a1_0 * x1S[t] + a2_0 * x2S[t] + a3_0 * x3S[t])))
        p11 = 1 / (1 + np.exp(-(a0_1 + a1_1 * x1S[t] + a2_1 * x2S[t] + a3_1 * x3S[t])))
        p01 = 1 - p00
        p10 = 1 - p11
        
        # Forecast of Y at period t under different regimes
        yf0 = const0 + beta1_0 * x1[t] + beta2_0 * x2[t] + beta3_0 * x3[t]
        yf1 = const1 + beta1_1 * x1[t] + beta2_1 * x2[t] + beta3_1 * x3[t]
        
        # State probabilities for period t
        p0 = lastObservedState[0] * p00 + lastObservedState[1] * p10
        p1 = lastObservedState[0] * p01 + lastObservedState[1] * p11
        
        # Redefine last observed state
        lastObservedState[0] = p0
        lastObservedState[1] = p1
        
        # Forecasts
        yf_w[t] = yf0 * p0 + yf1 * p1
        yf_top[t] = np.argmax([yf0, yf1])
        yf_r1[t] = yf0
        yf_r2[t] = yf1
    
    res = pd.DataFrame({'yf_w': yf_w,
                        'yf_top': yf_top,
                        'yf_r1': yf_r1,
                        'yf_r2': yf_r2})
    
    return res


def nowcast_error_endo_MS_2r(ddT, Xind=1, nLags=0, testLast=10, deleteLastX=0):
    aaa = ddT['Ydata'].isna()
    aaa[~aaa] = -999
    firstNArows = max(np.cumsum(aaa)) - 2

    if firstNArows > 0:
        ddT = ddT.iloc[firstNArows+1:, :]

    lastYRow = max(np.where(~ddT['Ydata'].isna())[0])
    nRowsInData = len(ddT)
    if nRowsInData - lastYRow > 3:
        ddT = ddT.iloc[:lastYRow + 4, :]

    lastYRow = max(np.where(~ddT['Ydata'].isna())[0])
    nRowsInData = len(ddT)

    lastDataRow = ddT.iloc[-1, :].copy()
    lastDataRow['Ydata'] = np.nan
    if nRowsInData - lastYRow == 2:
        ddT = ddT.append(lastDataRow, ignore_index=True)
    if nRowsInData - lastYRow == 1:
        ddT = ddT.append([lastDataRow, lastDataRow], ignore_index=True)
    if nRowsInData - lastYRow == 0:
        ddT = ddT.append([lastDataRow, lastDataRow, lastDataRow], ignore_index=True)

    APEm = []
    APEm_wFut = []
    APEm_topState = []
    APEm_best = []
    Yall = ddT['Ydata'].dropna().to_numpy()
    errTable = pd.DataFrame()

    ddT.iloc[:, Xind] = ddT.iloc[:, Xind].interpolate()

    for k in range(6, 3 * testLast + 1, 3):
        ddT2 = ddT.iloc[:-k, :]
        ddTest2 = ddT.iloc[-k:, :]

        thisXSeries = ddT.iloc[:, Xind]

        if deleteLastX > 0:
            thisXSeries[-(k // 3):] = np.nan
            thisXSeries = thisXSeries.fillna(method='ffill')

        X = ddT2.iloc[:, Xind].to_numpy()
        Xtest = ddTest2.iloc[:, Xind].to_numpy()
        Y = ddT2['Ydata'].dropna().to_numpy()

        tmp = pd.DataFrame(mls(X, k=range(3), m=3), columns=['X' + str(i) for i in range(3)]).assign(Y=Y)

        mLin = lm(Y ~ ., data=tmp)
        modMSwM = msmFit(mLin, k=2, p=nLags, sw=[True] * (5 + nLags), control={'parallel': False})

        if nLags == 0:
            nd = pd.DataFrame(mls(thisXSeries.to_numpy(), k=range(3), m=3), columns=['X' + str(i) for i in range(3)])
            nd.insert(0, 'ons', 1)
            ff = np.dot(nd.to_numpy(), modMSwM.Coef)
        else:
            nd = pd.DataFrame(mls(thisXSeries.to_numpy(), k=range(3), m=3), columns=['X' + str(i) for i in range(3)])
            nd.insert(0, 'ons', 1)
            for nl in range(1, nLags + 1):
                nd[f'lag{nl}'] = np.hstack([np.repeat(np.nan, nl + 1), Yall[:-(nl + 1)]])
            ff = np.dot(nd.to_numpy(), modMSwM.Coef)

        yf_point = ff
