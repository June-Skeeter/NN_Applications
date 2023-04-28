import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

def CI_Plot(ax,df,y,ci='CI95',linecolor='r',facecolor='#1356c240',edgecolor='#1356c2'):
    # Plot a line and shade its 95% Confidence interval
    ax.fill_between(df.index,df[y]-df['CI95'],df[y]+df['CI95'],facecolor=facecolor,edgecolor=edgecolor,label = '95% CI')
    ax.plot(df.index,df[y],color=linecolor,label = y)
    ax.grid()
    ax.set_ylabel(y)
    return(ax)

def Contour_Plot(fig,ax,grid_x1,grid_x2,grid_y,cmap='bwr',norm=None,unit='',bins=None):
    # Plot a colormesh grid with contours
    if norm is None:
        norm = Normalize(vmin=grid_y.min(), vmax=grid_y.max(), clip=False)
    elif len(norm)==3:
        norm = TwoSlopeNorm(vmin=norm[0], vcenter=norm[1], vmax=norm[2])
    else:
        norm = Normalize(vmin=norm[0], vmax=norm[1], clip=False)

    if bins is None:
        bins = np.linspace(grid_y.min(),grid_y.max(),10)
    
    def fmt(x):
        s = f"{x:.1f}"
        if s.endswith("0"):
            s = f"{x:.0f}"
        return rf"{s}"

    c = ax.pcolormesh(
        grid_x1,grid_x2,grid_y,
        cmap=cmap,
        norm=norm)
    cb = fig.colorbar(c)

    CS = ax.contour(
        grid_x1,grid_x2,grid_y,
        levels=bins,
        colors='k',
        )
    ax.clabel(CS, CS.levels[1::2], inline=True, fmt=fmt, fontsize=10)

    axcb = cb.ax
    xmin, xmax = ax.get_xlim()
    axcb.hlines(bins, xmin, xmax, colors=['black'])
    axcb.set_ylabel(unit)
    return(ax,axcb)