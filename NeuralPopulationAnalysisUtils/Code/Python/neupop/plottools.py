import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import warnings


def generate_color_arrays(n_colors, cmap_str='tab10', n_divs=10):
    cmap=cmx.get_cmap(cmap_str)
    colors = np.array([np.array(cmap(i)) for i in np.linspace(0, 1, max(n_colors,n_divs))])[0:n_colors]
    return colors

def add_alpha_to_color_arrays(color_arrays, alpha=0.3):
    return (color_arrays * np.array([1,1,1,alpha]))

def plot_w_error(y,x=None,ye=None,color=None,marker=None,legend=None,xlabel=None,ylabel=None,title=None, show_legend=True, ax=None):
    '''
    yerror is in the shape of n_data X errors or n_data X 2 X errors
    '''

    y = np.asarray(y)
    if y.size == 0:
        warnings.warn("Provided y vector is blank.",RuntimeWarning)
        return

    # n = number of rows (how many lines to plot)
    if len(y.shape) > 1:
        n = np.asarray(y).shape[0]
    else:
        n = 1
    y = y.reshape((n,-1))
    m = y.shape[1] # m = number of x points

    if x is None:
        x = range(m)
    x = np.asarray(x).reshape((-1,m))
    if x.shape[0] == 1:
        x = np.tile(x,[n,1])
    n_x = x.shape[0]
    assert n_x == n

    if ye is not None:
        ye = np.asarray(ye).reshape((n,-1,m))
        assert ye.shape[1] in [1,2]
        if ye.shape[1] == 1:
            ye_top = ye_bottom = ye[:,0,:]
        else: #ye.shape[1] == 2:
            ye_top = ye[:,0,:]
            ye_bottom = ye[:,1,:]

    if legend is not None:
        assert np.asarray(legend).size == n

    if color is not None:
        if isinstance(color, str):
            n_input_colors = 1
        else:
            n_input_colors = np.asarray(color).shape[0]
        colors = np.tile(color,(np.ceil(n/n_input_colors),1))
    else:
        colors = generate_color_arrays(n)
    error_colors = add_alpha_to_color_arrays(colors)

    if ax is None:
        _, ax = plt.subplots(1,1)

    for i in range(n):
        # Plot through nan trials to connect points with lines
        no_nan_idx = ~np.isnan(y[i])
        y_i = y[i][no_nan_idx]
        ye_top_i = ye_top[i][no_nan_idx]
        ye_bottom_i = ye_bottom[i][no_nan_idx]
        x_i = x[i][no_nan_idx]

        ax.fill_between(x_i,y_i+ye_top_i,y_i-ye_bottom_i, color=error_colors[i])
        if legend is not None:
            ax.plot(x_i,y_i,color=colors[i],marker=marker,label=legend[i])
        else:
            ax.plot(x_i,y_i,color=colors[i], marker=marker)

        #if marker is not None:
            #ax.plot(x_i,y_i+ye_top_i,color=error_colors[i],marker=marker)
            #ax.plot(x_i,y_i-ye_bottom_i,color=error_colors[i],marker=marker)

    if show_legend:
        ax.legend(loc='best')

    if title is not None:
        ax.title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return ax
