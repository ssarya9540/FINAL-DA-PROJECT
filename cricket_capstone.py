#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd 
import numpy as np


# # 1). Read csv data

# In[19]:


cricket=pd.read_csv(r'/Users/shubhamkumar/Desktop/cricket-2.csv',encoding='latin1')
cricket


# In[20]:


cricket.head()


# # 2). To check a size of data

# In[21]:


cricket.shape


# # 3). Information about data frame(columns, column labels, column data types, memory usage, range index,null,not null)

# In[22]:


cricket.info()


# # 4). Check in cricket data has null value or not null value

# In[23]:


cricket.isnull()


# #from this code we can identify the undefined column has null values in cricket data

# # 5). Now to Checking how many columns have null values 

# In[24]:


[col for col in cricket.columns if cricket[col].isnull().sum() > 0]


# #means in cricket data only one column has null values

# # 6). To drop the null values 

# In[25]:


unique_cols =  [x for x in cricket.columns if cricket[x].nunique()==1] 
print(unique_cols)
cricket.drop(unique_cols,axis=1,inplace=True)
cricket.columns


# In[26]:


cricket.drop('Remarks',axis=1,inplace=True)
cricket.head()


# #Here we deleted a all Null values from cricket data by using drop method and given inplace is true. Now we got a clear data to next preceed.

# In[27]:


cricket.head()


# #This is our not null(clear) data

# # 7). define statistical values of data using describe method(summery check)

# In[34]:


cricket.describe().T


# #here we defined the all summery of data

# # 8). To Data Transformation

# In[15]:


cricket[['Start', 'Last']] = cricket['Span'].str.split('-', n=1, expand=True)
cricket.head()


# #here we seperated the Span coloumn into two different coloumns start and Last

# In[16]:


cricket.info()


# #Here we can see the Start and Last coloumn are not in integer. 

# # 9). To convert Start and Last column from object to integer

# In[17]:


cricket[['Start','Last']]=cricket[['Start','Last']].astype(int)


# In[16]:


cricket.info()


# #Now the Start and Last coloumn are in integer type

# # 10). Add the coloumn in data

# In[17]:


cricket['Exp']=cricket['Last']-cricket['Start']
cricket


# #from this code we got a total year of play cricket for each player

# # 11). to delete not required columns

# In[18]:


cricket=cricket.drop(['Span','Start','Last'],axis=1)


# In[19]:


cricket.head()


# #Here we deleted the unnecessory column(Start,Last)

# In[20]:


x=cricket.pop('Exp')


# In[21]:


cricket.head()


# In[22]:


cricket.insert(2,'Exp',x)


# In[23]:


cricket.head()


# #here we change the actual position of coloumn Exp from index-2 position

# # 12). Visualization

# In[36]:


import matplotlib.pyplot as plt
import seaborn as sns


# ## Barplot

# In[37]:


# graph1)
plt.figure(figsize=(30,5))
mat=cricket[['Player','Mat']].sort_values('Mat',ascending=False)
ax=sns.barplot(x='Player',y='Mat',data=mat)
ax.set(xlabel='',ylabel='Match played')
plt.xticks(rotation=90)
plt.show()


# #plt.figure(figsize=(30,5)) This line sets the size of the chart.(30=30inch width and 5=5inch height)
# #mat=cricket[['Player','Mat']].sort_values('Mat',ascending=False) In this line we selected player and mat coloumns and coloumn Mat arranged into decending order and assigned it into new coloumn Mat.
# #ax=sns.barplot(x='Player',y='Mat',data=mat) in this line the 'Player' column is plotted on the x-axis and the 'Mat' column is plotted on the y-axis.
# #ax.set(xlabel='',ylabel='Match played') here we describe the lebel of y-axis='Match played' and x-axis= empty string.
# #plt.xticks(rotation=90)= here Rotates the x-axis labels by 90 degrees
# #plt.show()= at last display the result

# In[38]:


#*Result of that graph1)*= Sachin Tendulkar is the most match played batsman,and CG Greenidge is the less match played batsman.


# In[39]:


#graph2)
plt.figure(figsize=(30,5))
runs=cricket[['Player','Runs']].sort_values('Runs',ascending=False)
ax=sns.barplot(x='Player',y='Runs',data=runs)
ax.set(xlabel='',ylabel='runs')
plt.xticks(rotation=90)
plt.show()


# #*Result of graph2)*= Sachin Tendulkar is the most run scorer and Abdul Razzaq is the less run scorer in all matches.

# In[40]:


#graph3)
plt.figure(figsize=(30,5))
inns=cricket[['Player','Inns']].sort_values('Inns',ascending=False)
ax=sns.barplot(x='Player',y='Inns',data=inns)
ax.set(xlabel='',ylabel='INNINGS')
plt.xticks(rotation=90)
plt.show()


# #*Result of graph3)*=Sachin Tendulakar is the most innings played and CG Greenidge(WI) is the less innings played.

# In[41]:


#graph 4).
plt.figure(figsize=(30,5))
no=cricket[['Player','NO']].sort_values('NO',ascending=False)
ax=sns.barplot(x='Player',y='NO',data=no)
ax.set(xlabel='',ylabel='Not outs')
plt.xticks(rotation=90)
plt.show()


# #*Result of graph4)*= MS Dhoni is the not out in most of matches and Tamim Iqbal is not out in less of matches.

# In[42]:


cricket.head()


# In[43]:


cricket.info()


# In[44]:


cricket.HS=cricket.HS.str.extract('(\d+)')
cricket.HS=cricket.HS.astype(int)


# In[45]:


cricket.info()


# In[46]:


#graph 5)
#cricket.HS=cricket.HS.str.extract('(\d+)')
#cricket.HS=cricket.HS.astype(int)

plt.figure(figsize=(30,5))
HS=cricket[['Player','HS']].sort_values('HS',ascending=False)
ax=sns.barplot(x='Player',y='HS',data=HS)
ax.set(xlabel='',ylabel='High score')
plt.xticks(rotation=90)
plt.show()
plt.savefig('output.png')


# #*Result of graph 5)*=RG Sharma is the highest scorer of the perticular matches and Misbah-ul-Haq(Pak) is lowest score of the perticular matches. 

# ## Heatmap

# In[47]:


cricket.head()


# ## Heatmap

# In[52]:


plt.figure(figsize=(10,6))
sns.heatmap(cricket.corr(),annot=True,cmap='Blues')
plt.show()


# #Result of heatmap=Basically heatmap is use for the showing the correlation of data.

# In[38]:


cricket_drop=cricket.copy()
player=cricket_drop.pop('Player')


# In[39]:


cricket_drop


# #Here we delete the column player by using pop and save the data as n new variable cricket_drop 

# # 13). Model building

# ### Rescaling= #rescaling is a common data preprocessing technique that can help to make data more comparable, interpretable, and suitable for analysis or modeling

# In[45]:


import sklearn
from sklearn.preprocessing import StandardScaler


# In[46]:


scaler=StandardScaler()


# In[47]:


cricket_scaled=scaler.fit_transform(cricket_drop)


# In[48]:


cricket_scaled


# In[49]:


cricket.columns


# In[50]:


cricket_cricket1=pd.DataFrame(cricket_scaled,columns=['Mat', 'Exp', 'Inns', 'NO', 'Runs', 'HS', 'Ave', 'BF', 'SR',
       '100', '50', '0'])
cricket_cricket1.head()


# #here we created a new DataFrame with renamed columns and scaled data, which can be used for further analysis.

# ### Clustering 

# In[54]:


from sklearn.cluster import KMeans


# In[55]:


#elbow 
clusters=list(range(2,8))
ssd=[]
for num_clusters in clusters:
    model_clus=KMeans(n_clusters=num_clusters,random_state=50,max_iter=150)
    model_clus.fit(cricket_cricket1)
    ssd.append(model_clus.inertia_)
plt.plot(clusters,ssd);


# In[56]:


cluster=KMeans(n_clusters=4,random_state=15)
cluster.fit(cricket_cricket1)


# #here we defined clusters of our cricket data

# ### Silhouette score analysis to find the ideal number of clusters for K-means clustering

# In[57]:


#avg,sr,no


# In[58]:


from sklearn.metrics import silhouette_score


# In[59]:


# Silhouette score analysis to find the ideal number of clusters for K-means clustering

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50,random_state= 0)
    kmeans.fit(cricket_cricket1)
    
    cluster_labels = kmeans.labels_
    
    
    silhouette_avg = silhouette_score(cricket_cricket1, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))


# In[60]:


cluster_labels


# In[61]:


cricket['Cluster_Id']=cluster_labels
cricket


# ## Scatterplot

# In[63]:


plt.figure(figsize=(20,15))
plt.subplot(3,1,1)
sns.scatterplot(x='Ave',y='NO',hue='Cluster_Id',data=cricket,legend='full',palette='Set1')
plt.subplot(3,1,2)
sns.scatterplot(x='Ave',y='SR',hue='Cluster_Id',data=cricket,legend='full',palette='Set1')
plt.subplot(3,1,3)
sns.scatterplot(x='NO',y='SR',hue='Cluster_Id',data=cricket,legend='full',palette='Set1')
plt.show()


# In[64]:


cricket[cricket['Cluster_Id']==0].sort_values(by=['NO','Ave','SR'],ascending=[False,False,False]).head()


# In[65]:


cricket[cricket['Cluster_Id']==1].sort_values(by=['NO','Ave','SR'],ascending=[False,False,False]).head()


# In[66]:


cricket[cricket['Cluster_Id']==2].sort_values(by=['NO','Ave','SR'],ascending=[False,False,False]).head()


# In[67]:


cricket[cricket['Cluster_Id']==3].sort_values(by=['NO','Ave','SR'],ascending=[False,False,False]).head()


# In[68]:


cricket[cricket['Cluster_Id']==4].sort_values(by=['NO','Ave','SR'],ascending=[False,False,False]).head()


# In[69]:


cricket[cricket['Cluster_Id']==5].sort_values(by=['NO','Ave','SR'],ascending=[False,False,False]).head()


# In[70]:


cricket[cricket['Cluster_Id']==6].sort_values(by=['NO','Ave','SR'],ascending=[False,False,False]).head()


# In[71]:


cricket[cricket['Cluster_Id']==7].sort_values(by=['NO','Ave','SR'],ascending=[False,False,False]).head()


# In[ ]:


# Conclusion
SR Waugh (AUS)
MN Samuels (WI)
MS Dhoni (Asia/INDIA)
SR Tendulkar (INDIA)
MG Bevan (AUS)
Yuvraj Singh (Asia/INDIA)
Shahid Afridi (Asia/ICC/PAK)
AB de Villiers (Afr/SA)
These are the best batsman according to our model building respectively.

