# load CSV
# label encoder:
# counter each category
# from machine learning homework

def main():
# import csv data
    import pandas as pd

    wineData = pd.read_csv('../dat/motionDetect.csv')
    wineDataArray = wineData.values

    print(wineDataArray.shape)
    X = wineDataArray[:,:562]
    y = wineDataArray[:,562]

# Label Encoder:
# It is used to transform non-numerical labels to numerical labels (or nominal categorical variables).
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    #x = ['LAYING', 'STANDING', 'WALKING', 'SITTING', 'WALKING_UPSTAIRS']
    Y = label_encoder.fit_transform(y)

    from collections import Counter
    print(format(Counter(Y)))
    print(format(Counter(y)))

if __name__=="__main__":
    main()
    
