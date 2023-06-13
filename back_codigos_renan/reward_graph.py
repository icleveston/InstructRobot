import matplotlib, csv
matplotlib.rcParams["backend"] = "TkAgg"
from matplotlib import pyplot as plt
from os.path import join
import numpy as np
from itertools import islice

filexp = []

# Policies with partial variables in the reward.csv
#filexp.append("TestingAllBigger_2020_12_14_00_39_42_0000--s-0")

# Policies with all variables in the reward.csv
#filexp.append("TestingAllBigger_2020_10_13_00_21_19_0000--s-0")
#filexp.append("TestingAllBigger_2020_10_15_03_33_40_0000--s-0")
#filexp.append("TestingAllBigger_2020_12_07_04_16_13_0000--s-0")
#filexp.append("TestingAllBigger_2021_01_14_20_51_58_0000--s-0")
#filexp.append("TestingAllBigger_2021_01_15_02_35_36_0000--s-0")
#filexp.append("TestingAllBigger_2021_01_21_09_12_52_0000--s-0")
#filexp.append("TestingAllBigger_2021_02_10_11_03_34_0000--s-0")
#filexp.append("TestingAllBigger_2021_02_11_23_03_44_0000--s-0")
#filexp.append("TestingAllBigger_2021_02_13_03_57_22_0000--s-0")
#filexp.append("TestingAllBigger_2021_02_14_04_48_21_0000--s-0")
#filexp.append("TestingAllBigger_2021_02_18_04_54_06_0000--s-0")
#filexp.append("TestingAllBigger_2021_02_20_06_58_52_0000--s-0")
#filexp.append("TestingAllBigger_2021_02_20_22_25_33_0000--s-0")
#filexp.append("TestingAllBigger_2021_02_24_14_42_54_0000--s-0")
#filexp.append("TestingAllBigger_2021_02_27_03_47_40_0000--s-0")


filexp.append("TestingAllBigger_2021_04_16_18_35_33_0000--s-0")
filexp.append("TestingAllBigger_2021_04_16_19_11_37_0000--s-0")
#filexp.append("TestingAllBigger_2021_03_10_14_33_59_0000--s-0")

fileroot = "/home/nanbaima/Documents/naovrepenv/rlkit/data/TestingAllBigger/"

def plot_multivalue(values, columns, labels, title, mean, path):
#def plot_multivalue(values, columns, labels, title, mean, i):
    handles = []

    plt.clf()
#    plt.style.use('seaborn-white')
    for c, l in zip(columns, labels):
        v = [float(v[c]) for v in values]
        v_mean=np.mean(np.array(v[:len(v)-len(v)%mean]).reshape(-1, mean), axis=1)
        handles.append(plt.plot(range(len(v_mean)), v_mean, label=l)[0])

    plt.legend(handles=handles)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path + "/" + title + ".png", dpi=800)
    #plt.savefig(fileroot + "img/" + filexp[i] + ".png", dpi=800)
    # plt.show()
    # plt.close()

if __name__ == "__main__":
	for i in range(len(filexp)):
		filedic = join(fileroot, filexp[i])
		f = open(join(filedic,"reward.csv"))
		reader = csv.reader(f, delimiter=",")
		
		values = [v for v in reader]
		
		plot_multivalue(
		    #values[:2500000],
		    values,
		    [0, 1, 2,
		    3,
		    4,
		    5,
		    6,
		    7,
		    8, 
		    9,10,
		    11],
		    ["3 Collisions", "Touch", "No Touch",
		    "Object Grabbed",
            "Object Catching",
		    "Object Let Go",
		    "Object Raised",
		    "Object Raising",
		    "Object Dropped",
		    "Lef Hand Distance", "Right Hand Distance",
		    "Reward"],
		    "Declared Reward",
		    5000,
		    filedic)
		    #i)
