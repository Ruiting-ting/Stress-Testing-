import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tqdm import tqdm
import pandas as pd

class LocalLinearTrend  (sm.tsa.statespace.MLEModel):
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
        self.ssm['transition'] = np.array([[1, 1],[0, 1]])
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
        self.ssm['obs_cov',0,0] = params[0]

        # State covariance
        self.ssm[self._state_cov_idx] = params[1:]

def fnDataImport(bDropNA=True):

    df = pd.read_excel("D:\QRM_Program\Research_Project\MarketData.xlsx", parse_dates=['Quarter'], index_col='Quarter')
    df.columns = ["GDP", "WTI", "HPI", "SMX", "ASCX"]

    if bDropNA:
        return df.dropna()
    else:
        return df


def fnOU_calibration(mReturn, iSteps, dt):
    mParams = np.zeros(shape=(iSteps + 2, 5))
    for stock in range(mReturn.shape[1]):
        # Get the time-varying means;
        LLT = LocalLinearTrend(mReturn[:, stock][~np.isnan(mReturn[:, stock])]).fit(disp=False)
        predict = LLT.get_prediction()
        dMu = predict.predicted_mean

        a = np.empty((len(dMu)+12))
        a[:] = np.nan

        forecast = LLT.get_forecast(12)

        a[-12:] = forecast.predicted_mean

        plt.plot(dMu)
        plt.plot(mReturn[:, stock][~np.isnan(mReturn[:, stock])], linestyle='--')
        plt.hlines(np.mean(mReturn[:, stock][~np.isnan(mReturn[:, stock])]), xmin=0, xmax=len(dMu))
        plt.plot(a)
        plt.show()

        # AR(1) process for lamda and sigma;
        mod = sm.tsa.arima.ARIMA(mReturn[:, stock][~np.isnan(mReturn[:, stock])], order=(1, 0, 0))
        res = mod.fit()

        # return parameters
        alpha = res.params[1]
        beta = res.params[0]
        sd_epsilon = np.sqrt(res.params[2])

        # calculate parameters for OU model
        dLambda = -np.log(alpha) / dt
        dSigma = sd_epsilon * np.sqrt(-2*np.log(alpha) / dt*(1-alpha**2))

        mParams[0:12, stock] = forecast.predicted_mean
        mParams[12, stock] = dLambda
        mParams[13, stock] = dSigma

    return mParams



def fnOU_simulation(T, dt, iSims, mReturn, mCorr):
    """Short summary.

    Parameters
    ----------
    T : type
        Description of parameter `T`.
    dt : type
        Description of parameter `dt`.
    iSims : type
        Description of parameter `iSims`.
    mReturn : type
        Description of parameter `mReturn`.
    mCorr : type
        Description of parameter `mCorr`.

    Returns
    -------
    Matrix
        S - a (steps+1)-by-nsims-by-nassets 3-dimensional matrix where
        each row represents a time step, each column represents a
        seperate simulation run and each 3rd dimension represents a
        different asset.
    """
    iSteps = int(T/dt)

    # return calibrated parameters from function based on data given
    mParams = fnOU_calibration(mReturn, iSteps, dt)  # [dLambda, dMu, dSigma]
    vMu = mParams[0:iSteps,:]
    vLambda = mParams[iSteps , :]
    vSigma = mParams[iSteps + 1, :]

    # S_0 should be equal to the last observed variable
    S_0 = mReturn[-1]

    # calculate amount of steps in simulations
    mS = np.zeros([iSteps + 1, iSims, len(S_0)])
    mS[0, :, :] = S_0

    for sim in tqdm(range(iSims)):
        mDW = np.random.multivariate_normal(np.zeros(len(S_0)), mCorr, iSteps + 1)
        for i in range(1, iSteps + 1):
            part1 = mS[i-1, sim, :] * np.exp(-vLambda*dt) + vMu[i-1,:]*(1-np.exp(-vLambda*dt))
            mS[i, sim, :] =  part1 + vSigma*(np.sqrt((1-np.exp(-2*vLambda*dt))/(2*vLambda)))*mDW[i]

    return mS

def Main():
        # load all data and also changes
    dfFull = fnDataImport(bDropNA=False)

    # calculate log differences for additivity later on
    dfReturnFull = np.log(dfFull).diff()

    # clip extreme 1% and 99% values to better approximate "real-world"
    dfReturnFullClipped = dfReturnFull.copy()
    dfReturnFullClipped[['GDP', 'HPI']] = dfReturnFull[['GDP', 'HPI']].clip(lower=dfReturnFull['GDP'].quantile(0.01), upper=dfReturnFull['GDP'].quantile(0.99))

    # get historical 3 year changes/ 12 quarters
    mHistorical = dfFull.pct_change(12).values

    # calculate correlation matrix and changes matrix
    mCorr = dfReturnFull.corr().values
    mReturn = dfReturnFullClipped.values

    # magic numbers
    T = 12  # simulate 12 quarters ahead
    dt = 1  # each step is one quarter (20 working days)
    iSims = 1_000_000  # simulate iSim scenarios
    iSteps = int(T/dt) # time steps

    mParams = fnOU_calibration(mReturn, iSteps, dt)  # [dLambda, dMu, dSigma]
    vMu = mParams[0:iSteps,:]
    mS = fnOU_simulation(T, dt, iSims, mReturn, mCorr)

    fig = plt.figure(figsize=(10, 12))
    fig.suptitle('Simulating %i paths for %i assets' % (iSims, len(dfFull.columns)))
    columns = 2
    rows = 3
    list_assets = ["GDP", "WTI", "HPI", "SMX", "ASCX"]

    for i in range(1, 6):
        fig.add_subplot(rows, columns, i)
        plt.hist([np.sum(mS[-12:, :, i-1], axis=0), mHistorical[:, i-1]], color=['g', 'r'],
                 label=['Generated 3Y-change Asset '+list_assets[i-1],
                        'Historical 3Y-change Asset '+list_assets[i-1]], bins=40, density=True)
        plt.legend()
    plt.show()

    #print(mReturn[:, 0][~np.isnan(mReturn[:, 0])])
    print(vMu)

    #plt.plot(mReturn[:, 0][~np.isnan(mReturn[:, 0])], label='Observations')
    #plt.plot(vMu[:, 0], label='One-step-ahead Prediction')
    #plt.legend()
    #plt.show()

if __name__ == '__main__':
    Main()
