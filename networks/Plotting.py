import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate(i):
    x.append(i)
    y.append(R[i])

    ax.clear()
    ax.plot(x, y)
    #ax.axhline(1000,color='b')
    ax.set_xlim([0,300])
    ax.set_ylim([0,100000])
    ax.legend('Current Balance','Starting Balance')

x=[]
y=[]
R=[]
with open('20220329_RNNTraining.txt') as f:
    lines = f.readlines()
for i in range(len(lines)):
    R.append(float(lines[i]))

#print(lines)
#print(R)
fig, ax = plt.subplots()


ani = FuncAnimation(fig,animate,frames=300,interval=500,repeat=False)
plt.show()
