import numpy as np
from collections import defaultdict

def k_nearest_neighbours(features, labels, to_be_predicted, k=3):
    if len(np.unique(labels)) >= k:
        print("The value of k is less than the number of groups")
        return -1
    
    distance_array = []
    for point, group in zip(features, labels):
        distance = np.sqrt(np.sum((point - to_be_predicted)**2))
        distance_array.append((distance, group))

    candidate_neighbours = [i[1] for i in sorted(distance_array)[:k]]

    candidate_neighbours_dict = defaultdict(int)

    for item in candidate_neighbours:
        if item in candidate_neighbours_dict:
            candidate_neighbours_dict[item] += 1
        else:
            candidate_neighbours_dict[item] = 1

    max_item = None
    max_item_count = 1

    for item in candidate_neighbours_dict:
        if candidate_neighbours_dict[item] >= max_item_count:
            max_item = item

    return (max_item)

if __name__ =="__main__":
    dataset = np.array([
        [1,2,1],
        [2,3,1],
        [1,3,1],
        [9,8,2],
        [9,10,2],
        [10,12,2]
    ])
    to_be_predicted = np.array([6,5])

    resultant_group = k_nearest_neighbours(dataset[:,[0,1]], dataset[:, 2], to_be_predicted, k=3)
    if resultant_group != -1:
        print(f"The point {to_be_predicted} belongs to group {resultant_group}")