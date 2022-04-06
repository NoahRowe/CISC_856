import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate(i):
    x.append(i)
    y.append(R[i])

    plt.clear()
    plt.plot(x, y)
    plt.axhline(1000,color='b')
    plt.xlim([0,200])
    plt.ylim([0,100000])
    plt.legend('Current Balance','Starting Balance')
    


x=[]
y=[]
R=[]


with open('20220329_RNNTraining.txt') as f:
    lines = f.readlines()
for i in range(len(lines)):
    R.append(float(lines[i]))

#print(lines)
# print(len(R))
# fig = plt.figure(figsize=((10,8)))
#ani = FuncAnimation(fig,animate,frames=200,interval=100)
#ani.save(r'RewardPlot.gif')

for i in range(200):
    x.append(i)
    y.append(R[i])
    plt.clf()
    plt.xlim(0,200)
    plt.ylim(0,max(R))
    plt.pause(0.001)

plt.show()

