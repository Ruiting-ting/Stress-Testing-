#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
import scipy.optimize as opt


def fnDataImport(bDropNA=True):
    """Short summary.

    Parameters
    ----------
    bDropNA : type
        Description of parameter `bDropNA`.

    Returns
    -------
    type
        Description of returned object.

    """

    df = pd.read_excel("MarketData.xlsx", parse_dates=['Quarter'], index_col='Quarter')
    df.columns = ["GDP", "WTI", "HPI", "SMX", "ASCX"]

    if bDropNA:
        return df.dropna()
    else:
        return df


class LocalLinearTrend(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        # Model order
        k_states = k_posdef = 2

        # Initialize the statespace
        super(LocalLinearTrend, self).__init__(
            endog, k_states=k_states, k_posdef=k_posdef,
            initialization='approximate_diffuse',
            loglikelihood_burn=k_states
        )

        # Initialize the matrices
        self.ssm['design'] = np.array([1, 0])
        self.ssm['transition'] = np.array([[1, 1], [0, 1]])
        self.ssm['selection'] = np.eye(k_states)

        # Cache some indices
        self._state_cov_idx = ('state_cov',) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        return ['sigma2.measurement', 'sigma2.level', 'sigma2.trend']

    @property
    def start_params(self):
        return [np.std(self.endog)]*3

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super(LocalLinearTrend, self).update(params, *args, **kwargs)

        # Observation covariance
        self.ssm['obs_cov', 0, 0] = params[0]

        # State covariance
        self.ssm[self._state_cov_idx] = params[1:]


class MultivariateSimulation:
    def __init__(self, iT, dt, iSims, mPrices):
        self.iT = iT
        self.dt = dt
        self.iSteps = int(self.iT/self.dt)

        self.iSims = iSims
        self.mPrices = mPrices
        self.iAssets = mPrices.shape[1]

    # Auxiliary Functions
    def _calculateLogReturns(self, Winsorize):
        if Winsorize:
            df = np.log(self.mPrices).diff()
            df['GDP'] = df['GDP'].clip(lower=df['GDP'].quantile(
                0.05), upper=df['GDP'].quantile(0.95))
            return df
        else:
            df = np.log(self.mPrices).diff()
            return df

    def _calculateSimpleReturns(self):
        return (self.mPrices).diff()

    def _calculateCorrMatrix(self, mReturn):
        return mReturn.corr()

    # Calibration

    def _fnGBM_calibration(self, mReturn, KF=False):
        iSteps = 12
        mParams = np.zeros(shape=(self.iSteps+2, ))
        dMean = np.mean(mReturn[~np.isnan(mReturn)])
        dVar = np.var(mReturn[~np.isnan(mReturn)])
        # put parameters in mParams array
        mParams[0:self.iSteps] = np.repeat(dMean, self.iSteps)

        # Get the time-varying means;
        if KF:
            LLT = LocalLinearTrend(mReturn[~np.isnan(mReturn)]).fit(disp=False)
            predict = LLT.get_prediction()
            dMu = predict.predicted_mean

            forecast = LLT.get_forecast(12)

            a = np.empty((len(dMu)+12))
            a[:] = np.nan
            a[-12:] = forecast.predicted_mean
            mParams[0:self.iSteps] = forecast.predicted_mean
        # calculate mean
        #dVar = np.var(mReturn[~np.isnan(mReturn)])
        # put parameters in mParams array
        mParams[-2] = dVar
        return mParams

    def _fnABM_calibration(self, mReturn):
        mParams = np.zeros(shape=(2, 5))
        for stock in range(mReturn.shape[1]):
            if OLS:
                endog = mReturn[:, stock][~np.isnan(mReturn[:, stock])]
                exog = np.ones(len(endog))
                model = sm.OLS(endog, exog).fit()
                dMean = model.params[0]
                dSigma = np.std(model.resid)
            else:
                dMean = np.mean(mReturn[:, stock][~np.isnan(mReturn[:, stock])])
                dSigma = np.std(mReturn[:, stock][~np.isnan(mReturn[:, stock])])
            # put parameters in mParams array
            mParams[:, stock] = np.array([dMean, dSigma])

        return mParams

    def _fnOU_calibration(self, mReturn, KF=False):
        mParams = np.zeros(shape=(self.iSteps+2, ))
        # mParams = np.zeros(shape=(3, 1))

        # fit ARIMA(1,0,0) model > AR(1)
        mod = sm.tsa.arima.ARIMA(mReturn[~np.isnan(mReturn)], order=(1, 0, 0))
        res = mod.fit()

        # return parameters
        alpha = res.params[1]
        beta = res.params[0]
        sd_epsilon = np.sqrt(res.params[2])

        # calculate parameters for OU model
        dLambda = -np.log(alpha) / self.dt
        dMu = beta/(1-alpha)
        dSigma = sd_epsilon * np.sqrt(-2*np.log(alpha) / self.dt*(1-alpha**2))

        # put parameters in mParams array
        # mParams = np.array([dLambda, dMu, dSigma])
        mParams[0:self.iSteps] = np.repeat(dMu, self.iSteps)

        if KF:
            LLT = LocalLinearTrend(mReturn[~np.isnan(mReturn)]).fit(disp=False)
            Predict = LLT.get_prediction()
            vMuPredict = Predict.predicted_mean
            vMu = LLT.get_forecast(12)

            a = np.empty((len(vMuPredict)+12))
            a[:] = np.nan
            a[-12:] = vMu.predicted_mean

            # plt.plot(vMuPredict)
            # plt.plot(mReturn[~np.isnan(mReturn)], linestyle='--')
            # plt.hlines(np.mean(mReturn[~np.isnan(mReturn)]), xmin=0, xmax=len(vMuPredict))
            # plt.plot(a)
            # plt.show()

            mParams[0:self.iSteps] = vMu.predicted_mean

        mParams[-2] = dLambda
        mParams[-1] = dSigma

        return mParams

    def _fnMJD_calibration(self, vTheta, vReturn):

        # starting values: d is for the diffusion part, j is for the jumps
        mu_d = vTheta[0]
        sigma_d = np.exp(vTheta[1])
        mu_j = vTheta[2]
        sigma_j = vTheta[3]
        Lambda = vTheta[4]

        for k in range(0, 20):
            mean = (mu_d - sigma_d**2 / 2 * self.dt) + (mu_j * k)
            std = (sigma_d ** 2 * self.dt + sigma_j**2 * k)
            xpdfs = norm.pdf(x=vReturn, loc=mean, scale=np.sqrt(std))

            pk_denominator = (Lambda * self.dt) ** k / np.math.factorial(k)
            Pk = (pk_denominator * np.exp(-Lambda*self.dt))
            obj = - np.sum(np.log(xpdfs + Pk))

        return obj

    def _calibrateModels(self, sModel, mReturn):
        if sModel == 'GBM':
            mParams = self._fnGBM_calibration(mReturn)
        elif sModel == 'ABM':
            mParams = self._fnABM_calibration(mReturn)
        elif sModel == 'OU':
            mParams = self._fnOU_calibration(mReturn)
        elif sModel == 'MJD':
            mParams = np.zeros((5, self.iAssets))
            vTheta0 = [0.01, 2, 0.08, 0.02, 2]
            res = opt.minimize(self._fnMJD_calibration, vTheta0, args=(
                mReturn[~np.isnan(mReturn)]), method='L-BFGS-B')
            mParams = res.x

        return mParams

    # Simulation

    def fnGBM_simulation(self, mCorr, mParams, mRandomDraws, idx):
        # return calibrated parameters from function based on data given
        vMu = mParams[:12]
        vSigma = mParams[12]

        # pre-allocate the output
        mS = np.zeros([self.iSteps + 1, self.iSims])
        mS[0, :] = self.mPrices.values[-1, idx]
        # generate correlated random sequences and paths
        for sim in tqdm(range(self.iSims)):
            # generate correlated random sequence
            mDW = mRandomDraws[:, sim]
            for i in range(1, self.iSteps+1):
                mS[i, sim] = mS[i-1, sim] * np.exp((vMu[i-1]*self.dt) + (np.sqrt(vSigma) * np.sqrt(self.dt)*mDW[i]))
        return mS


    def fnABM_simulation(self, mCorr, mParams):
        # return calibrated parameters from function based on data given
        vMu = mParams[0, :]
        vSigma = mParams[1, :]

        # pre-allocate the output
        mS = np.zeros([self.iSteps + 1, self.iSims, self.iAssets])
        mS[0, :, :] = self.mPrices.values[-1, :]
        # generate correlated random sequences and paths
        for sim in tqdm(range(self.iSims)):
            # generate correlated random sequence
            mDW = np.random.multivariate_normal(np.zeros(self.iAssets), mCorr, self.iSteps + 1)
            for i in range(1, self.iSteps+1):
                mS[i, sim, :] = mS[i-1, sim, :] + vMu*self.dt + vSigma * np.sqrt(self.dt) * mDW[i]
        return mS

    def fnOU_simulation(self, mCorr, mParams, mReturn, mRandomDraws, idx):
        # return calibrated parameters from function based on data given
        vMu = mParams[0:self.iSteps]
        vLambda = mParams[self.iSteps]
        vSigma = mParams[self.iSteps + 1]

        # pre-allocate the output
        mS = np.zeros([self.iSteps + 1, self.iSims])
        mS[0, :] = mReturn[-1]

        for sim in tqdm(range(self.iSims)):
            mDW = mRandomDraws[:, sim]

            for i in range(1, self.iSteps + 1):
                mS[i, sim] = mS[i-1, sim] * np.exp(-vLambda*self.dt) + vMu[i-1]*(1-np.exp(-vLambda*self.dt)) +                     vSigma*(np.sqrt((1-np.exp(-2*vLambda*self.dt))/(2*vLambda)))*mDW[i]
        return np.sum(mS[-12:, :], axis=0)

    def fnMJD_simulation(self, mCorr, mParams, mRandomDraws, idx):
        vMu = mParams[0]
        vSigma = np.exp(mParams[1])
        vJumps_mu = mParams[2]
        vJumps_sigma = mParams[3]
        vLambdas = mParams[4]

        decomposition = np.linalg.cholesky(mCorr)

        # pre-allocate the output
        mS = np.zeros([self.iSteps + 1, self.iSims])
        mS[0, :] = self.mPrices.values[-1, idx]

        for sim in tqdm(range(self.iSims)):
            # Z_1 = np.random.normal(0., 1., size=(self.iSteps + 1, self.iAssets))
            # Z_2 = np.random.normal(0., 1., size=(self.iSteps + 1, self.iAssets))
            Poisson = np.random.poisson(lam=vLambdas*self.dt, size=(self.iSteps + 1, 1))

            # Z_1_1 = Z_1 @ decomposition
            # Z_2_1 = Z_2 @ decomposition
            Z_1_1 = mRandomDraws[:, sim]
            Z_2_1 = mRandomDraws[:, sim] * np.random.normal(1, 0.1)
            for i in range(1, self.iSteps + 1):
                musigmaDelta = (vMu - vSigma**2/2) * self.dt
                sigmasqrtDelta = vSigma * np.sqrt(self.dt)

                expPar1 = musigmaDelta + sigmasqrtDelta * Z_1_1[i]
                expPar2 = vJumps_mu * Poisson[i, :] + vJumps_sigma *                     np.sqrt(Poisson[i, :]) * Z_2_1[i]

                mS[i, sim] = mS[i - 1, sim] * np.exp(expPar1 + expPar2)

        return mS

    def fnSimulate(self, listModel, listAsset, sName, Winsorize, sSave):
        mReturn = self._calculateLogReturns(Winsorize=Winsorize)

        mCorr = self._calculateCorrMatrix(mReturn).values

        mRandomDraws = np.random.multivariate_normal(
            mean=np.zeros(self.iAssets), cov=mCorr, size=(self.iSteps + 1, self.iSims))
        mReturn = mReturn.values
        if len(listModel) == 1:
            listModel = 5*listModel
        else:
            listModel = listModel

        mSimulatedPrices = np.zeros([self.iSteps + 1, self.iSims, self.iAssets])
        mSimulatedDistributions = np.zeros((self.iSims, self.iAssets))

        for idx, model in enumerate(listModel):

            print("Using %s model for asset %s" % (listModel[idx], listAsset[idx]))
            if model == 'GBM':
                mParams = self._calibrateModels(model, mReturn[:, idx])
                mS = self.fnGBM_simulation(mCorr, mParams, mRandomDraws[:, :, idx], idx)
                mSimulatedDistributions[:, idx] = ((mS[-1, :] - mS[0, :]) / mS[0, :])
                mSimulatedPrices[:, :, idx] = mS

            # # elif model == 'ABM':
            #     mReturn = self._calculateSimpleReturns()
            #     mCorr = self._calculateCorrMatrix(mReturn).values
            #     mReturn = mReturn.values
            #     mParams = self._calibrateModels(model, mReturn)
            #     mS = self.fnABM_simulation(mCorr, mParams)
            #     mSimulatedDistributions = np.zeros((self.iSims, self.iAssets))
            #     for i in range(self.iAssets):
            #         mSimulatedDistributions[:, i] = ((mS[-1, :, i] - mS[0, :, i]) / mS[0, :, i])

            elif model == 'OU':
                mParams = self._calibrateModels(model, mReturn[:, idx])
                mSimulatedDistributions[:, idx] = self.fnOU_simulation(
                    mCorr, mParams, mReturn[:, idx], mRandomDraws[:, :, idx], idx)

            elif model == 'MJD':
                mParams = self._calibrateModels(model, mReturn[:, idx])
                mS = self.fnMJD_simulation(
                    mCorr, mParams, mRandomDraws[:, :, idx], idx)
                mSimulatedDistributions[:, idx] = ((mS[-1, :] - mS[0, :]) / mS[0, :])
                mSimulatedPrices[:, :, idx] = mS

        if sSave:
            print("Saving file...")
            np.savetxt("%s.csv" % sName,
                       mSimulatedDistributions, delimiter=",", fmt='%.6e')
        else:
            print("File has not been saved")

        return mSimulatedDistributions, mSimulatedPrices, listModel

    # Plot Histograms

    def fnPlotHistograms(self, listModel, listAsset, sName, Winsorize, sSave):
        mSimulatedDistributions, mSimulatedPrices, listModel = self.fnSimulate(
            listModel, listAsset, sName, Winsorize, sSave)

        mHistorical = self.mPrices.pct_change(12).values

        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('Simulating %i paths for %i assets using %s model' %
                     (self.iSims, self.iAssets, listModel))
        columns = 2
        rows = 3
        for i in range(1, 6):
            fig.add_subplot(rows, columns, i)
            plt.hist([mSimulatedDistributions[:, i-1], mHistorical[:, i-1]], color=['g', 'r'],
                     label=['Generated 3Y-change Asset '+listAsset[i-1]+' by model '+listModel[i-1],
                            'Historical 3Y-change Asset '+listAsset[i-1]], bins=40, density=True)
            plt.legend()
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        return mSimulatedDistributions, mSimulatedPrices, listModel


def Main():
    # load all data and also changes
    dfFull = fnDataImport(bDropNA=False)

    iT = 12
    dt = 1
    iSims = 500
    mPrices = dfFull
    #listModel = ['GBM']
    listAsset = ["GDP", "WTI", "HPI", "SMX", "ASCX"]

    cMS = MultivariateSimulation(iT=iT, dt=dt, iSims=iSims,
                                 mPrices=mPrices)

    listModels = [['GBM']]
    for listModel in listModels:
        mSimulatedDistributions, mSimulatedPrices, listModel = cMS.fnPlotHistograms(
            listModel=listModel, listAsset=listAsset, sName='-'.join(listModel), Winsorize=True, sSave=True)

    # mHistorical = dfFull.pct_change(12)
    # np.savetxt("sims/Historical.csv", mHistorical, delimiter=",", fmt='%.6e')


if __name__ == '__main__':
    Main()


# In[ ]:
