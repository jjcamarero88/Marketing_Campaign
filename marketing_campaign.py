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
