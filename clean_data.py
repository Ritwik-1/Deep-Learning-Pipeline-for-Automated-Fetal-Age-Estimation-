import pandas as pd 

df = pd.read_csv(r"FETAL_PLANES_ZENODO\FETAL_PLANES_DB_data.csv",sep=';')

df = df[['Image_name','Patient_num','Plane']]
df = df[df['Plane'] != 'Other']
df = df[df['Plane'] != 'Maternal cervix']
df = df[df['Plane'] != 'Fetal thorax']

# only kept these classes 
# {'Fetal abdomen', 'Fetal brain', 'Fetal femur'}

# df.columns = df.columns.str.strip()  
print(set(df["Plane"].tolist()))
print("Length of the dataframe : ",df.shape)

df.to_csv("Filtered_data.csv",sep = ";")

