#%%

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

print('imported.')

df = pd.read_csv('./erp.csv')
# lst = df.values.tolist()

df_bus = df['Build-business']
lst = df_bus.dropna().tolist()

df_pb = df['Pilot-business']
lst.extend(df_pb.dropna().tolist())

df_ob = df['Others-business']
lst.extend(df_ob.dropna().tolist())

df_bi = df['Build-it']
lst.extend(df_bi.dropna().tolist())

df_pi = df['Pilot-it']
lst.extend(df_pi.dropna().tolist())

df_oi = df['Other-it']
lst.extend(df_oi.dropna().tolist())


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(lst)

k = 7
model = KMeans(n_clusters=k, max_iter=1000)
model.fit(X)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),

print("\n")
print("Prediction")

Y = vectorizer.transform(["new data"])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["new case"])
prediction = model.predict(Y)
print(prediction)

output = model.predict(X)
df_out = pd.DataFrame({'sentence': lst, 'class':output})

#%%

di = {0: 'require', 1: 'pilot', 2: 'business', 3: 'data', 4: 'test', 5:'coe', 6:'user'}
df_out = df_out.replace({'class': di})


df_out.to_csv('./out.csv')
