def plot_histogram_with_descriptive_stats(data, std_multiple=1):
    """Plots a histogram of the given data with the mean and standard deviation shown

    Parameters:
    ----------
    data: pd.Series
        a pandas series that contains the data
    """
    mean_value = data.mean()
    std_value = data.std()

    plot_axis = data.plot.hist()
    y_min, y_max = plot_axis.get_ylim()

    plot_axis.axvline(x=mean_value, color='red', linestyle='dashed')
    plot_axis.errorbar(mean_value, (y_min + y_max) / 2, xerr=std_multiple*std_value, color='green')