from matplotlib import pyplot as plt


def histAllColumns(data, columns, directory):
    for column in columns:
        print('plotting: ', column)
        plt.figure()
        plt.hist(data[column], bins=10)
        plt.title(column)
        plt.savefig(directory + '/' + column + '.png')
        plt.close()