import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.colors import rgb2hex
import matplotlib.cm as cm
import plotly.express as px
import plotly.graph_objects as go
import squarify
from plotly.offline import init_notebook_mode,iplot
from wordcloud import WordCloud
import numpy as np 
from scipy import stats
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.colors 
from collections import Counter
import datetime as dt
cmap2 = cm.get_cmap('twilight',13)
colors1= []
for i in range(cmap2.N):
    rgb= cmap2(i)[:4]
    colors1.append(rgb2hex(rgb))
    #print(rgb2hex(rgb))

# Set style
sns.set(style='whitegrid')

#read csv
df = pd.read_csv("marketing_campaign.csv",sep='\t')

#info about csv
print(df.info())

#drop missing values as they are 1%
df = df.dropna()
print(df.info())
print(df.head())

#describe df 
print(df.describe())


#Get unique values of Education Column
print(df["Education"].unique())

#Count the number of participants by Education
plt.figure(figsize=(12,6))
p_education = sns.countplot(data=df,x="Education",palette="crest",linewidth=1,edgecolor="black", order = df['Education'].value_counts().index)
plt.xlabel("Education")
plt.ylabel("Count")
plt.tight_layout()
plt.title("Count Participants by Education")
plt.show()

#Get unique values of Marital_Status
print(df["Marital_Status"].unique())

#Separete DF using Year_Birth
print(df["Year_Birth"].unique())

#Separate the DF in decades.
period_length = 10
start_year = 1893
end_year = 1996
year_range = end_year - start_year
modulo = year_range % period_length
print(modulo)

# Next, let’s find the starting and ending years for our last period. The addition of one is done to include the last year as well
if modulo == 0:
    final_start = end_year - period_length
else:
    final_start = end_year - modulo
final_end = end_year + 1

#Use NUMPY to create a list of the earlier starting years for the range
starts = np.arange(start_year, final_start, period_length).tolist()

#create lists of tuples, where each tuple is like (period_start, period_end). From these tuples we can finally create our bins as Pandas IntervalIndex.
tuples = [(start, start+period_length) for start in starts]
tuples.append(tuple([final_start, final_end]))
bins = pd.IntervalIndex.from_tuples(tuples, closed='left')

#These bins then convert to labels nicely by converting them to string. I created a dictionary for easy replacement in the DataFrame.
original_labels = list(bins.astype(str))
new_labels = ['{} - {}'.format(b.strip('[)').split(', ')[0], int(b.strip('[)').split(', ')[1])-1) for b in original_labels]
label_dict = dict(zip(original_labels, new_labels))

# Assign each row to a period. Then with Pandas cut(), we can easily place the content of year column into those bins and create a new column ‘PERIOD’. Finally, the bin labels are replaced with help of the label_dict.
df['Period'] = pd.cut(df['Year_Birth'], bins=bins, include_lowest=True, precision=0)
df['Period'] = df['Period'].astype("str")
df = df.replace(label_dict)

#Count the number of participants by Period
plt.figure(figsize=(12,6))
p_period = sns.countplot(data=df,x="Period",palette="mako",linewidth=1,edgecolor="black")
plt.xlabel("Period")
plt.ylabel("Count")
plt.tight_layout()
plt.title("Count Participants by Period")
plt.show()

#Bar plot customers by Period with Response
plt.figure(figsize=(12,6))
bar_period = sns.barplot(data=df,x="Period",y="Response",palette="crest",linewidth=1,edgecolor="black")
plt.xlabel("Period")
plt.ylabel("Response")
plt.tight_layout()
plt.title("Bar plot of Customers by Period")
plt.show()

# Customer's Education profile and Period rate
plt.figure(figsize = (12,6))
sns.barplot(x='Education',y='Response',data=df,palette=colors1, hue='Period')
plt.xlabel('Education Qualifications')
plt.title("Customer Education Profile")
plt.tight_layout()
plt.show()

# Customer's Education profile and Marital Status rate
plt.figure(figsize = (12,6))
sns.barplot(x='Education',y='Response',data=df, palette=colors1, hue='Marital_Status')
plt.xlabel('Education Qualifications')
plt.title("Customer Education Profile")
plt.tight_layout()
plt.show()

# Marital Status Vs Response Rate
plt.figure(figsize=(15,6))
sns.histplot( x="Marital_Status", data=df, hue="Response",stat="percent", multiple="stack",palette='mako')
plt.grid(False)
plt.show()

# Marital Status Vs Response Rate = 1
plt.figure(figsize=(15,6))
sns.histplot( x="Marital_Status", data=df[df["Response"] == 1],stat="percent", multiple="stack",palette='mako')
plt.grid(False)
plt.show()


# Year Birth Vs Response Rate
plt.figure(figsize=(15,6))
sns.histplot( x="Year_Birth", data=df,hue="Response", stat="percent", multiple="stack",palette='crest')
plt.grid(False)
plt.show()

# Year Birth Vs Response Rate = 1
plt.figure(figsize=(15,6))
sns.histplot( x="Year_Birth", data=df[df["Response"] == 1], stat="percent", multiple="stack",palette='crest')
plt.grid(False)
plt.show()

# Customer's Relationship profile and Complain
plt.figure(figsize = (12,6))
sns.barplot(x='Marital_Status',y='Complain', data=df,palette=colors1)
plt.xlabel('Relationship Profile')
plt.title("Customer Relationship and Complain Profile")
plt.tight_layout()
plt.show()

# Customer's Period and Complain
plt.figure(figsize = (12,6))
sns.barplot(x='Period',y='Complain', data=df,palette=colors1)
plt.xlabel('Relationship Profile')
plt.title("Customer Relationship and Complain Profile")
plt.tight_layout()
plt.show()

# Most bought products
pr = df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]
pr_means = pr.mean(axis=0).sort_values(ascending=False)
pr_means_df = pd.DataFrame(list(pr_means.items()), columns=['Product', 'Avg Spending'])
plt.figure(figsize=(12,6))
plt.title('Avg Spending on the different Products')
sns.barplot(data=pr_means_df, x='Product', y='Avg Spending',palette='mako');
plt.xlabel('Product Names', fontsize=20, labelpad=20)
plt.ylabel('Av Spending', fontsize=20, labelpad=20)
plt.show()

#Avg Spending by Period
avg_spending_period = df.groupby("Period")[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].mean()
avg_spending_period.plot(kind='bar',figsize=(12, 6))
plt.title('Average Spending per Period')
plt.xlabel('Period')
plt.ylabel('Average Spending')
plt.xticks(rotation=45)
plt.show()

#Number of purchases by Period
plt.figure(figsize=(12,6))
sns.barplot(data=df,x="Period",y="NumDealsPurchases",palette=colors1)
plt.xlabel('Period')
plt.title("Number of Purchases")
plt.tight_layout()
plt.show()

# Total Purchase/Spendings on diferent products by Income
df['Total_purchase'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts']+df['MntSweetProducts']+df['MntGoldProds']
plt.figure(figsize=(12,6))
sns.scatterplot(x=df['Income'], y=df['Total_purchase'],s=150)
plt.grid(False)
plt.xlabel('Customer Income')
plt.ylabel('Total Purchase/Spendings')
plt.show()

# Total Purchase by Marital Status
plt.figure(figsize=(10, 5),dpi=100)
sns.barplot(x='Marital_Status',y='Total_purchase',data=df,palette=colors1)
plt.xlabel('Marital Status')
plt.ylabel('Total Purchase/Spendings')
plt.show()

# Total Purchase by Period
plt.figure(figsize=(10, 5),dpi=100)
sns.barplot(x='Period',y='Total_purchase',data=df,palette="mako")
plt.xlabel('Period of Birth')
plt.ylabel('Total Purchase/Spendings')
plt.show()


#Calculate Age of each Customer
df['Age'] = 2024 - df["Year_Birth"]

# Histplot by Age
plt.figure(figsize=(30, 8))
plt.title('Age distribution')
ax = sns.histplot(df['Age'].sort_values(), bins=45)
plt.xticks(np.linspace(df['Age'].min(), df['Age'].max(), 45, dtype=int, endpoint = True))
plt.grid(False)
plt.show()

#Income distribution comparing Complain using a continous distribution.
plt.figure(figsize=(15,7))
sns.kdeplot(
   data=df, x="Income", hue="Complain", log_scale= True,
   fill=True, common_norm=False,palette='crest',
   alpha=.5, linewidth=0,
)
plt.xlabel('Income')
plt.show()

#Income distribution comparing Kids at home using a continous distribution.
plt.figure(figsize=(15,7))
sns.kdeplot(
   data=df, x="Income", hue="Kidhome", log_scale= True,
   fill=True, common_norm=False,palette='mako',
   alpha=.5, linewidth=0,
)
plt.xlabel('Income')
plt.show()

#HeatMap 
# Dropping two constant variables Z_CostContact and Z_Revenue
df_drop = df.drop(['Z_CostContact', 'Z_Revenue'], axis=1)

# Selecting only numeric columns
numeric_df = df_drop.select_dtypes(include=[float, int])

# HeatMap plot
df_corr = numeric_df.corr()
plt.figure(figsize=(20, 18)) 
sns.heatmap(data=df_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True, linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Grouping Education 
df['Education']=df['Education'].str.replace('Graduation','Higher Education')
df['Education']=df['Education'].str.replace('PhD','Higher Education')
df['Education']=df['Education'].str.replace('Master','Higher Education')
df['Education']=df['Education'].str.replace('2n Cycle','Higher Education')

# Goruping Marital Status
df['Marital_Status']=df['Marital_Status'].str.replace('Married','In A Relationship')
df['Marital_Status']=df['Marital_Status'].str.replace('Together','In A Relationship')
df['Marital_Status']=df['Marital_Status'].str.replace('Divorced','Single')
df['Marital_Status']=df['Marital_Status'].str.replace('Widow','Single')
df['Marital_Status']=df['Marital_Status'].str.replace('Alone','Single')
df['Marital_Status']=df['Marital_Status'].str.replace('Absurd','Single')
df['Marital_Status']=df['Marital_Status'].str.replace('YOLO','Single')

# Grouping Kids
df['Total_child']=df['Kidhome']+df['Teenhome']

# Campaign
df['Camp_total']=df['AcceptedCmp1']+df['AcceptedCmp2']+df['AcceptedCmp3'] +df['AcceptedCmp4']+df['AcceptedCmp5'] +df['Response']

# Removing Outliers
df=df.loc[np.abs(stats.zscore(df['Income']))<3]
df.reset_index(inplace=True)
df=df.drop(columns=['index'])  
print(df.shape)

# Label Encoding the Data (Education and Marital Status Column)
cols=["Age","Education", "Marital_Status","Income", "Camp_total", 'Total_child','Total_purchase']
c_df=df[cols]
l=LabelEncoder()
c_df['Education']=c_df[['Education']].apply(l.fit_transform)
c_df['Marital_Status']=c_df[['Marital_Status']].apply(l.fit_transform)
# Standard Scaling
ss=StandardScaler()
c_df_final=ss.fit_transform(c_df)

#Optimum no of Clusters
l1=[]
for i in range(1,13):
    k_mean=KMeans(n_clusters=i,random_state=32,init="k-means++")
    k_mean.fit(c_df_final)
    l1.append(k_mean.inertia_)
plt.plot(range(1,13),l1)
plt.scatter(range(1,13),l1,color="red")
plt.show()

# Kmeans (4 Clusters)
km=KMeans(n_clusters=4,random_state=0,init="k-means++")
km.fit(c_df_final)
clusters=km.predict(c_df_final)
c_df['cluster_no'] = clusters

#plot
plt.figure(figsize=(10,7))
plt.scatter(df['Income'],df['Total_purchase'],c=clusters, cmap='icefire')
plt.xlabel('Income')
plt.ylabel('Total Purchase')
plt.grid(False)
plt.show()

#print number of customers under each cluster
print(c_df['cluster_no'].value_counts())

#Print the spending and Income mean by Cluster
print("Cluster 0 Total Spending: ", c_df.loc[c_df['cluster_no']== 0 ,['Total_purchase'] ].mean()['Total_purchase'])
print("Cluster 1 Total Spending: ",c_df.loc[c_df['cluster_no']== 1 ,['Total_purchase'] ].mean()['Total_purchase'])
print("Cluster 2 Total Spending: ",c_df.loc[c_df['cluster_no']== 2 ,['Total_purchase'] ].mean()['Total_purchase'])
print("Cluster 3 Total Spending: ",c_df.loc[c_df['cluster_no']== 3 ,['Total_purchase'] ].mean()['Total_purchase'])
print("Cluster 0 Income: ",c_df.loc[c_df['cluster_no']== 0 ,['Income'] ].mean()['Income'])
print("Cluster 1 Income: ",c_df.loc[c_df['cluster_no']== 1 ,['Income'] ].mean()['Income'])
print("Cluster 2 Income: ",c_df.loc[c_df['cluster_no']== 2 ,['Income'] ].mean()['Income'])
print("Cluster 3 Income: ",c_df.loc[c_df['cluster_no']== 3 ,['Income'] ].mean()['Income'])


# Kmeans (5 Clusters)
km=KMeans(n_clusters=5,random_state=0,init="k-means++")
km.fit(c_df_final)
clusters=km.predict(c_df_final)
c_df['cluster_no'] = clusters

#plot
plt.figure(figsize=(10,7))
plt.scatter(df['Income'],df['Total_purchase'],c=clusters, cmap='icefire')
plt.xlabel('Income')
plt.ylabel('Total Purchase')
plt.grid(False)
plt.show()

#print number of customers under each cluster
print(c_df['cluster_no'].value_counts())

#Print the spending and Income mean by Cluster
print("Cluster 0 Total Spending: ", c_df.loc[c_df['cluster_no']== 0 ,['Total_purchase'] ].mean()['Total_purchase'])
print("Cluster 1 Total Spending: ",c_df.loc[c_df['cluster_no']== 1 ,['Total_purchase'] ].mean()['Total_purchase'])
print("Cluster 2 Total Spending: ",c_df.loc[c_df['cluster_no']== 2 ,['Total_purchase'] ].mean()['Total_purchase'])
print("Cluster 3 Total Spending: ",c_df.loc[c_df['cluster_no']== 3 ,['Total_purchase'] ].mean()['Total_purchase'])
print("Cluster 4 Total Spending: ",c_df.loc[c_df['cluster_no']== 4 ,['Total_purchase'] ].mean()['Total_purchase'])
print("Cluster 0 Income: ",c_df.loc[c_df['cluster_no']== 0 ,['Income'] ].mean()['Income'])
print("Cluster 1 Income: ",c_df.loc[c_df['cluster_no']== 1 ,['Income'] ].mean()['Income'])
print("Cluster 2 Income: ",c_df.loc[c_df['cluster_no']== 2 ,['Income'] ].mean()['Income'])
print("Cluster 3 Income: ",c_df.loc[c_df['cluster_no']== 3 ,['Income'] ].mean()['Income'])
print("Cluster 4 Income: ",c_df.loc[c_df['cluster_no']== 4 ,['Income'] ].mean()['Income'])
