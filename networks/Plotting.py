
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

even = 6E9
x_data=[]
y_data=[]
R=[]
colors = ['g']
bounds = [0]

with open('networks/20220329_RNNTraining.txt') as f:
    lines = f.readlines()

for i in range(len(lines)):
    R.append(float(lines[i]))


for i in range(1,len(R)):
    if R[i]<R[i-1]:
        if colors[-1] != 'r':
            colors.append('r')
            bounds.append(i)
    else:
        if colors[-1] != 'g':
            colors.append('g')
            bounds.append(i)
    
if bounds[-1] != 199:
    bounds.append(199)

#print(bounds)
#print(colors)
cmap = ListedColormap(colors)

fig, ax = plt.subplots()
ax.clear()
#ax.axhline(even,color='b')
ax.set_xlim(0,200)
ax.set_ylim(0,max(R))
#ax.legend('Current Balance','Starting Balance')
#line, = ax.plot(0,0)


norm = BoundaryNorm(bounds, cmap.N)
line = LineCollection([], cmap=cmap,norm=norm)
line.set_array(np.linspace(0,199,200))
ax.add_collection(line)



def animate(i):
    x_data.append(i)
    y_data.append(R[i])

    points = np.array([x_data, y_data]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    line.set_segments(segments)

    # line.set_xdata(x_data)
    # line.set_ydata(y_data)
    # if R[i]<R[i-1]:
    #     line.set_color('red')
    # else:
    #     line.set_color('green')
    
    return line,
    

ani=FuncAnimation(fig,func=animate,frames=np.arange(0,199,1),interval=100,repeat=False)
ani.save(r'TestAnimation2.gif')

plt.show()

