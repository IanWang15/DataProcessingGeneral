# several methods to deal with imbalanced data
# from machine learning homework
    
wineData = pd.read_csv('../dat/winequality-red.csv')
wineDataArray = wineData.values

print(wineDataArray.shape)
X = wineDataArray[:,:11]
Y = wineDataArray[:,11]
    
#    deal with imbalance data

# under sampling
# all category numbers reduce to the minimal category
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_resampled, Y_resampled = rus.fit_resample(X, Y)
print(format(Counter(Y_resampled)))


# over sampling
# simple repeat data in the some small categories
# auto is ratio 1:1

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0, sampling_strategy='auto')
X_resampled, Y_resampled = ros.fit_resample(X, Y)


# over sampling
# all category numbers increase to the maximal category

from imblearn.over_sampling import SMOTE
sos = SMOTE(random_state=0)
X_resampled, Y_resampled = sos.fit_resample(X, Y)
print(format(Counter(Y_resampled)))

# over sampling then under sampling
# all category numbers increase to slightly lower than the maximal category

from imblearn.combine import SMOTETomek
kos = SMOTETomek(random_state=0) #
X_resampled, Y_resampled = kos.fit_sample(X, Y)
print(format(Counter(Y_resampled)))
