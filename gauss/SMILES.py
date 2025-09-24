# The SMILES strings

import pandas as pd

# 设置文件路径
file_path = r"D:\Python\pycharm\Barrier_Prediction\descriptors\gauss\Hf_TS_XYZs.csv"


df_TS = pd.read_csv(file_path)

# 查看 DataFrame 的前几行，确认数据已正确加载
print(df_TS.head())


R1 = df_TS.iloc[:,1]
R2 = df_TS.iloc[:,2]
R3 = df_TS.iloc[:,3]

print(df_TS['R1'].unique())
print(df_TS['R2'].unique())
print(df_TS['R3'].unique())


ligand_dict = {
    'CF3': 'C(F)(F)F',
    'Cl': 'Cl',
    'F': 'F',
    'OMe': 'OC',
    'Me': 'C',
    'Et': 'CC',
    'NO2': '[N+]([O-])=O',
    'NCH3_2': 'N(C)C'
}

SMILES = list()

for i in range(len(R1)):
    smi = 'Cl[Hf]12345678(C9C1(C2C3C94[R1])[R2])(C%10C5C6C7C%108[R3])Cl'
    if df_TS.loc[i].R1 == 'Ph':
        smi = smi.replace('[R1]', 'C%11=CC=CC=C%11')
    else:
        smi = smi.replace('[R1]', ligand_dict[df_TS.loc[i].R1])
    if df_TS.loc[i].R2 == 'Ph':
        smi = smi.replace('[R2]', 'C%11=CC=CC=C%11')
    else:
        smi = smi.replace('[R2]', ligand_dict[df_TS.loc[i].R2])
    if df_TS.loc[i].R3 == 'Ph':
        smi = smi.replace('[R3]', 'C%10=CC=CC=C%10')
    else:
        smi = smi.replace('[R3]', ligand_dict[df_TS.loc[i].R3])
    SMILES.append(smi)

print("Number of Structures: ", len(SMILES))
