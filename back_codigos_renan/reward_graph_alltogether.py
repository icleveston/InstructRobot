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
filexp.append("TestingAllBigger_2021_04_06_03_08_50_0000--s-0") #GraspCube
filexp.append("TestingAllBigger_2021_04_14_06_03_09_0000--s-0") #GraspCylinder
filexp.append("TestingAllBigger_2021_04_16_19_11_37_0000--s-0") #GrapsSphere
#filexp.append("TestingAllBigger_2021_04_06_03_13_09_0000--s-0") #TouchCube
#filexp.append("TestingAllBigger_2021_04_14_23_05_59_0000--s-0") #TouchCylinder
#filexp.append("TestingAllBigger_2020_12_07_04_16_13_0000--s-0") #TouchSphere

fileroot = "/home/nanbaima/Documents/naovrepenv/rlkit/data/TestingAllBigger/"

def plot_multivalue(values, columns, labels, mean, path, i):
#def plot_multivalue(values, columns, labels, title, mean, i):
    handles = []

    #plt.clf()
    #plt.style.use('seaborn-white')
    v = [float(v[11]) for v in values]
    v_mean=np.mean(np.array(v[:len(v)-len(v)%mean]).reshape(-1, mean), axis=1)
    plt.plot(range(len(v_mean)), v_mean)[0]

    plt.legend(labels)
    #plt.title(title)
    #plt.tight_layout()
    #plt.savefig(path + "/" + title + ".png", dpi=800)
    #plt.savefig(fileroot + "img/" + filexp[i] + ".png", dpi=800)
    # plt.show()
    # plt.close()

if __name__ == "__main__":
    #plt.clf()
	for i in range(len(filexp)):
		filedic = join(fileroot, filexp[i])
		f = open(join(filedic,"reward.csv"))
		reader = csv.reader(f, delimiter=",")
		
		values = [v for v in reader]
		
		plot_multivalue(
		    values[:2500000],
		    [#0, 1, 2,
		    #3,
		    #4,
		    #5,
		    #6,
		    #7,
		    #8, 
		    #9,10,
		    11],
		    ["Grasp Cube", "Grasp Cylinder",
             "Graps Sphere"], #["Touch Cube",
             #"Touch Cylinder", "Touch Sphere"],
		    #"Declared Reward",
		    5000,
		    filedic,
		    i)
    
#plt.title("Declared Reward")
#plt.tight_layout()
plt.savefig(fileroot + "img/" + "GraspAffordanceReward" + ".png", dpi=800)
#plt.savefig(fileroot + "img/" + filexp[i] + ".png", dpi=800)
# plt.show()
# plt.close()
