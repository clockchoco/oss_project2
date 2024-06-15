import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

data = pd.read_csv('data/ratings.dat', sep='::', header=None, engine='python')
data.columns = ['userId', 'movieId', 'rating', 'trash']
d = data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
kmeans = KMeans(n_clusters=3)
kmeans.fit(d)
users_index = kmeans.predict(d)
u1 = d[users_index == 0]
u2 = d[users_index == 1]
u3 = d[users_index == 2]

def print_top10_results(arr, algorithm, group_num):
    print("Top 10 Recommendation Results(MovieID) ", end="")
    print(f"for GROUP{group_num}")
    print(f"Algorithm: [{algorithm}]")
    for i, movie_id in enumerate(arr.index, start=1):
        print(f"Top {i}: {movie_id}")
    print()
def au(user_group, group_num):
    sum_result = user_group.sum()
    sum_result_sorted = sum_result.sort_values(ascending=False)
    # print(sum_result_sorted.head(10))
    print_top10_results(sum_result_sorted.head(10), algorithm="Additive Utilitarian", group_num=group_num)
def avg(user_group, group_num):
    user_group = user_group.replace(0, np.nan)
    avg_result = user_group.mean()
    avg_result_sorted = avg_result.sort_values(ascending=False)
    # print(avg_result_sorted.head(10))
    print_top10_results(avg_result_sorted.head(10), algorithm="Average", group_num=group_num)
def sc(user_group, group_num):
    user_group = user_group.replace(0, np.nan)
    sc_result = user_group.count()
    sc_result_sorted = sc_result.sort_values(ascending=False)
    # print(sc_result_sorted.head(10))
    print_top10_results(sc_result_sorted.head(10), algorithm="Simple Count", group_num=group_num)
def av(user_group, group_num):
    user_group = user_group.replace(0, np.nan)
    av_result = user_group[user_group > 3].count()
    av_result_sorted = av_result.sort_values(ascending=False)
    # print(av_result_sorted.head(10))
    print_top10_results(av_result_sorted.head(10), algorithm="Approval Voting", group_num=group_num)
def bc(user_group, group_num):
    user_group = user_group.replace(0, np.nan)
    bc_result = user_group.rank(axis=1, method="average", ascending=True) - 1
    bc_result_sum = bc_result.sum()
    # print(bc_result_sum)
    bc_result_sorted = bc_result_sum.sort_values(ascending=False)
    # print(bc_result_sorted.head(10))
    print_top10_results(bc_result_sorted.head(10), algorithm="Borda Count", group_num=group_num)
def cr(user_group, group_num):
    n_users, n_items = user_group.shape
    user_group_filled = user_group.replace(0, -np.inf)
    cr_result = np.zeros((n_items, n_items))

    for i in range(n_items):
        for j in range(i + 1, n_items):
            comp = np.sign(user_group_filled.iloc[:, j] - user_group_filled.iloc[:, i])
            score = np.sign(np.sum(comp))
            cr_result[i, j] = score
            cr_result[j, i] = -score

    cr_result_frame = pd.DataFrame(cr_result, columns=user_group.columns, index=user_group.columns)
    cr_result_sum = cr_result_frame.sum()
    cr_result_sorted = cr_result_sum.sort_values(ascending=False)

    print(cr_result_sorted.head(10))
    print_top10_results(cr_result_sorted.head(10), algorithm="Copeland Rule", group_num=group_num)

functions = [au, avg, sc, av, bc, cr]
groups = [(u1, "1"), (u2, "2"), (u3, "3")]
for func in functions:
    for group, num in groups:
        func(group, num)
