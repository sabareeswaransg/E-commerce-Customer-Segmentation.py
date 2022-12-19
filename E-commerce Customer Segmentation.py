import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

full_data = pd.read_csv('cust_data.csv')
print(full_data.columns)

# full_data.info()
x = full_data.head(100)

# Groping

a = input("Enter the brand name : ")

df = x.drop(['Gender', 'Orders'], axis=1)
kmeans = KMeans(n_clusters=3)
kmeans.fit(df)

plt.scatter(df['Cust_ID'], df[a], c=kmeans.labels_)
plt.show()

# Plotting the L-blow Graph

g = []
h = []

for i in range(1, 11):
    km = KMeans(n_clusters=i)
    km.fit_predict(df)
    g.append(km.inertia_)
    h.append(i)
plt.plot(g, h)
plt.show()
