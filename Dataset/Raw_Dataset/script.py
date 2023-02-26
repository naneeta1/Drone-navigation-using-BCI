import pandas as pd
import os

cwd = os.getcwd()

folders = ['back' , 'down' , 'forward' , 'left' , 'right' , 'up']

for fname in folders:
    paths = os.path.join(cwd,fname)
    #counter = 0
    for file_name_path in os.listdir(paths):
        file_name = os.path.join(paths,file_name_path)
        #file_name_path = file_name_path [-4:0]
        out_name = os.path.join(paths,file_name_path+".csv")
        with open(file_name, 'r') as fin:
            data = fin.read().splitlines(True)
        with open(file_name, 'w') as fout:
            fout.writelines(data[4:])
            
        df = pd.read_csv(file_name)

        indexSample=0
        count = 0
        for indexx, row in df.iterrows():
            if(row['Sample Index'] == 0):
                count=count+1
                if(count==11):
                    break
            else:
                indexSample=indexx

        df.drop(df.loc[0:indexSample].index, inplace=True)

        df.drop(df.index[0:indexSample])

        df = df[0:2560]
        #df.drop(df.columns[0], axis=1, inplace=True)

        #df.drop(df.iloc[:, 4:], axis=1, inplace=True)

        df.to_csv(file_name, header=False, index=False)

        #counter=counter+1