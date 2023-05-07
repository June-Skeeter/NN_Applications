import numpy as np
from sklearn import metrics
from matplotlib.colors import Normalize, TwoSlopeNorm

def makeRI_plot(ax,RI,Title=None,linecolor='r',facecolor='#1356c240',edgecolor='#1356c2'):
    ax.barh(RI.index,RI['RI_bar'],facecolor=facecolor,edgecolor=edgecolor,label='Mean RI')
    ax.errorbar(RI['RI_bar'],RI.index,xerr=RI['RI_CI95'],ecolor=linecolor,capsize=2,color='None',label = '95% CI')
    ax.grid()
    ax.set_ylabel('Inputs')
    ax.set_xlabel('RI%')
    ax.legend()
    ax.set_xscale('log')
    if Title is not None:
        ax.set_title(Title)

def make1_1_Plot(ax,df,x,y,unit='',linecolor='r',facecolor='#1356c240',edgecolor='#1356c2'):
    ax.scatter(df[x],df[y],color=facecolor,edgecolor=edgecolor)
    ax.plot(df[x],df[x],color=linecolor,label='1:1')
    r2 = np.round(metrics.r2_score(df[x],df[y]),2)
    RMSE = np.round(metrics.mean_squared_error(df[x],df[y])**.5,3)
    ax.set_title(f'RMSE = {RMSE} {unit} ;  $r^2$ = {r2}')
    ax.grid()
    ax.legend()
    return(ax)

def CI_Plot(ax,df,y,ci='Auto',linecolor='r',facecolor='#1356c240',edgecolor='#1356c2'):
    # Plot a line and shade its 95% Confidence interval
    if ci == 'Auto':
        ci = y+'_CI95'
    ax.fill_between(df.index,df[y]-df[ci],df[y]+df[ci],facecolor=facecolor,edgecolor=edgecolor,label = '95% CI')
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