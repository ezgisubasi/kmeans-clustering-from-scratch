"""
Name: Ezgi Nihal Subasi
Date: 29.03.2021
Project: K-Means Clustering

"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
import numpy as np


# implements kmeans using scikit learn library
def kmeans_scikit(df, k):
    # n_clusters takes number of clusters, init chooses random data points for the initial centroids
    # in default scikit provides 10 times and chooses the best one, to prevent that n_init assigned as 1
    model = KMeans(n_clusters=k, init='random', n_init=1)
    model.fit_transform(df)
    centroids = model.cluster_centers_ # final centroids
    # colors for 3, 4 and 6 k values
    rgb_colors = {0.: 'y',
                  1.: 'c',
                  2.: 'fuchsia',
                  }
    if k == 4:
        rgb_colors[3.] = 'lime'
    if k == 6:
        rgb_colors[3.] = 'lime'
        rgb_colors[4.] = 'orange'
        rgb_colors[5.] = 'tomato'

    new_labels = pd.Series(model.labels_.astype(float)) # label that predicted by kmeans

    plt.scatter(df['x'], df['y'], c=new_labels.map(rgb_colors), s=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=10)
    plt.xlabel('Final Cluster Centers\n Iteration Count=' +str(model.n_iter_)+
               '\n Objective Function Value: ' +str(model.inertia_))
    plt.ylabel('y')
    plt.title("k-Means")
    plt.show()


# implements kmeans from scratch
def kmeans_from_scratch(df, k, data_num):
    x_list, y_list = generate_random_points(df, k) # chooses different random points from dataset for initial clusters

    df_copy = df.copy() # copying data set
    check = True
    objective_function = [] # stores all object function values in each iteration
    iter = 0 # number of iteration
    points_colors = {} # colors for the clusters
    while (check):
        i = 0
        # calculates distance for every point according to its cluster centroid
        df_copy, total_points, obj_func = calculate_dist(df_copy, x_list, y_list, k)

        # sum of objective function values
        total_obf = 0
        for obj in obj_func.values():
            total_obf += obj

        # adds current objective function value to the list
        objective_function.append(total_obf)

        # stops the while loop when objective function starts to converge
        if abs(objective_function[iter - 1] - objective_function[iter]) < 0.01 and iter >= 1:
            check = False

        # calculates new values for centroids
        for coordinates in total_points.values():
            new_x, new_y = calculate_mean(coordinates)
            x_list[i] = new_x
            y_list[i] = new_y
            i += 1

        # plots the initial cluster centers
        if iter == 0:
            points_colors = initial_cluster_centers_plot(df_copy, x_list, y_list, k, data_num)

        iter += 1

    plot_objective_function(objective_function) # plots the objective function
    final_cluster_centers_plot(df_copy, x_list, y_list, points_colors, iter, objective_function[-1]) # plots final


# calculates euclid distance for two points
def euclid_calculator(x1, y1, x2, y2):
    distance = np.sqrt(np.square(abs(x1 - x2)) + np.square(abs(y1 - y2))) # euclid distance formula
    return distance


def calculate_dist(df_dist, x_list, y_list, k):
    obj_func = {} # stores object functions for each distances
    total_points = {} # stores all data points for each cluster centroids
    # creates points with names according to number of k
    for x in range(1, k + 1):
        name = 'point_'
        obj_func[name + str(x)] = 0
        total_points[name + str(x)] = []

    for index, row in df_dist.iterrows():
        points_distance = {}
        for x in range(1, k+1):
            name = 'point_'
            points_distance[name + str(x)] = 0
        for x, y, point in zip(x_list, y_list, points_distance.keys()):
            distance = euclid_calculator(x, y, row['x'], row['y']) # calculates distance for determine point is close to which center
            df_dist.loc[index, point] = distance # stores points with distances
            points_distance[point] = distance

        # sorts for finding every point close to which center
        sorted_dist = sorted(points_distance.items(), key=lambda kv: kv[1])
        point = [row['x'], row['y']]
        point_type = sorted_dist[0][0]
        obj_func[point_type] += np.square(sorted_dist[0][1])
        df_dist.loc[index, 'point_type'] = point_type
        total_points[point_type].append(tuple(point))

    return df_dist, total_points, obj_func


# calculates min x and y values for finding the new centroids
def calculate_mean(dist_list):
    x_total = 0
    y_total = 0
    for item in dist_list:
        x_total += item[0]
        y_total += item[1]

    mean_x = x_total / len(dist_list)
    mean_y = y_total / len(dist_list)

    return mean_x, mean_y


# calculates sum of all distances
def total_dist(dist):
    dist_sum = 0
    for item in dist:
        dist_sum += item
    return dist_sum


# plots objoctive function
def plot_objective_function(obj_func):
    x_axis = list(range(1, len(obj_func) + 1))
    # plotting the points
    plt.plot(x_axis, obj_func, marker='o')
    # naming the x axis
    plt.xlabel('Iteration')
    # naming the y axis
    plt.ylabel('Objective Function Value')
    # giving a title to my graph
    plt.title('Objective Function')
    # function to show the plot
    plt.show()


# reading csv file according to path and adds column names for separating class, x and y values
def reading_data(path):
    data = pd.read_csv(path)  # reading csv file
    first_row = []
    # appending first row in a list
    for i in range(0, 3):
        first_row.append(float(data.columns[i]))

    data.loc[-1] = first_row  # adding a row
    data.index = data.index + 1  # shifting index
    data = data.sort_index()  # sorting indexes

    data.columns = ["x", "y", "class"]  # assigning column names
    return data


# visualizing scatter plot version of data
def scatter_plot(plotted_df):
    fig, ax = plt.subplots()
    # selecting x and y axises without labels
    ax.scatter(plotted_df['x'], plotted_df['y'], c='y', s=0.5)
    plt.xlabel('x')  # entering name for x axis
    plt.ylabel('y')  # entering name for y axis
    plt.title("Dataset")  # entering a title for plot
    plt.show()


# plots the initial cluster centers
def initial_cluster_centers_plot(initial_df, x_list, y_list, k, data_num):
    fig, ax = plt.subplots()
    # defining colors for each clustered points at the beginning according to k number
    points_colors = {'point_1': 'y',
                     'point_2': 'c',
                     'point_3': 'fuchsia',
                     }
    if k == 4:
        points_colors['point_4'] = 'lime'
    if k == 6:
        points_colors['point_4'] = 'lime'
        points_colors['point_5'] = 'orange'
        points_colors['point_6'] = 'tomato'

    # selecting x and y axises with point type of each row
    ax.scatter(initial_df['x'], initial_df['y'], c=initial_df['point_type'].map(points_colors), s=0.5)
    # draws the new centroids according to mean values
    for x, y in zip(x_list, y_list):
        plt.scatter(x, y, c='black', s=10)

    plt.xlabel('x \n Initial Cluster Centers \n Dataset='+str(data_num)+'\n k='+str(k))
    plt.ylabel('y')
    plt.title("k-Means")
    plt.show()

    return points_colors


# plots the final cluster centers
def final_cluster_centers_plot(final_df, x_list, y_list, points_colors, iter_num, final_obf):
    fig, ax = plt.subplots()

    ax.scatter(final_df['x'], final_df['y'], c=final_df['point_type'].map(points_colors), s=0.5)

    for x, y in zip(x_list, y_list):
        plt.scatter(x, y, c='black', s=10)

    plt.xlabel('Final Cluster Centers\n Iteration Count=' +str(iter_num)+ '\n Objective Function Value:' +str(final_obf))
    plt.ylabel('y')
    plt.title("k-Means")
    plt.show()


# generates random points for initial cluster centroids
def generate_random_points(dataset, k):
    rand_x, rand_y = [], []
    random_points = random.sample(range(1, len(dataset) - 1), k)
    for rand_index in random_points:
        rand_x.append(dataset.iloc[rand_index][0])
        rand_y.append(dataset.iloc[rand_index][1])
    return rand_x, rand_y


if __name__ == "__main__":

    for i in range(1, 4):
        path = './data' + str(i) + '.txt'
        dataset = reading_data(path)
        unlabelled_data = dataset[['x', 'y']]

        scatter_plot(unlabelled_data)

        if i == 1:
            kmeans_from_scratch(unlabelled_data, 3, i)
            kmeans_scikit(unlabelled_data, 3)
            kmeans_from_scratch(unlabelled_data, 6, i)
            kmeans_scikit(unlabelled_data, 6)
        else:
            kmeans_from_scratch(unlabelled_data, 4, i)
            #kmeans_scikit(unlabelled_data, 4)



















