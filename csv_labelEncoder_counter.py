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

    
    
def loaddat(filename):
# another method to load csv

# initializing the titles and rows list
    fields = []
    var = []

    # reading csv file
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        fields = next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            var.append(row)

        # get total number of rows
#        print("Total no. of rows: %d"%(csvreader.line_num))
#    printing the field names
#    print('Field names are:' + ', '.join(field for field in fields))
    var0 = np.array(var).astype(float)
    return var0
    
if __name__=="__main__":
    main()
    
