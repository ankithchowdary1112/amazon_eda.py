import pandas as pd
import matplotlib.pyplot as plt
import seaborn as srn 


# To read CSV file
lk = pd.read_csv(r"c:\Users\Ankit\Downloads\amazon_products_sales_data_cleaned.csv")  # r -> 
print(lk.head())   # head() -> TO get top 5 details 

# complete info about the data set 
print(lk.info())

# To get count null values 
print(lk.isnull().sum())

# length of columns
print(len(lk))

# total percentage on null values 
print(lk.isnull().sum()*100 /len(lk))

# drop null value column sustainabilty tags (has 92% of null values )


# duplicate values in data set 
print(lk.duplicated().sum())


#  Action Items 

# total columns 
print(lk.columns)

# describe 

print(lk.describe().T)

# To get only number values 
num_df = lk.select_dtypes(include="number")

cat_df = lk.select_dtypes(exclude="number")

# To replace null values to medican values 

for column in num_df.columns:
   num_df[column] =  num_df[column].fillna(num_df[column].median())


print(num_df.isnull().sum())

#----------------------------------

# to remove null values with mode values in categorical columns

for column in cat_df.columns:
  mode_values = cat_df[column].mode().values[0]
  cat_df[column] =  cat_df[column].fillna(mode_values)

print(cat_df.isnull().sum())


# TO combine/concat
lk = pd.concat([num_df,cat_df] ,  axis= 1)
print(lk.head())

print(lk.isnull().sum())


# Day 2

# TO converrt the data types (float to int )

lk['total_reviews']= lk['total_reviews'].astype(int)
lk['purchased_last_month'] = lk['purchased_last_month'].astype(int)

print(lk.info())

# To convert STR / object to date and time 

lk['delivery_date' ] = pd.to_datetime(lk['delivery_date'], errors='coerce')
lk['data_collected_at' ] = pd.to_datetime(lk['data_collected_at'], errors='coerce')

print(lk.info())


# to drop column 

lk.drop(['sustainability_tags'], axis=1, inplace=True)

print(lk.info())


# Feature Engineering  -> using  existing column to create new column 

#discounted price

lk['discountes_price'] = lk['original_price'] - lk['discounted_price']

print(lk['discountes_price'])


# last month revenue 
lk['last_month_revenue'] = lk['purchased_last_month'] *lk['discounted_price']

print(lk['last_month_revenue'])



# which product has more discount and less discount

lk['discount_range'] = pd.cut(x= lk['discount_percentage'],
                              bins=[0 ,10,40,70,99], 
                              labels=['less discount','medium discount','high discount','very high'])
print(lk.head(2))


# Univariate Analysis 

   # categorical column

print(lk['discount_range'].value_counts())


# Numerical column

print(lk['total_reviews'].describe())  # complete information (how many values, mean, min % , max reviews )


#----------------------------------------------------------------


# Bi-variate analysis 

   # One categorical column and one numercical column   (Group by )
print( lk.groupby(['is_sponsored'])['purchased_last_month'].mean()
)

# with bar chart
print( lk.groupby(['is_sponsored'])['purchased_last_month'].sum().plot(kind= 'bar', figsize=(5,2)))
plt.show()


print(lk.groupby(['product_category'])['purchased_last_month'].sum().plot(kind='bar'))
plt.show()

# Category vs category 
 # convert to table format (frequency table )  (Crosstab)

print(lk.columns)

print(lk[['product_category','is_sponsored']])

# To create table format
print(pd.crosstab(index=lk['product_category'], columns=lk['is_sponsored']))



# Numerical vs Numerical  (Correlation )

 # correlation  ->  (dep on y - axis) indept on x-axis 


print(lk[['product_rating','purchased_last_month']].corr())

#-------------------------------------------------



# Multi variate analysis   (>2 columns )

 # Pivot table ->  index  (category), columns  (category) , values  (numberical)

print(lk[['product_category','is_sponsored','purchased_last_month']].head())



print(pd.pivot_table(lk, index=['product_category'], columns=['is_sponsored'], values=['purchased_last_month','product_rating'],aggfunc='sum'))