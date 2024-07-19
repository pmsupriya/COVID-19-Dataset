#!/usr/bin/env python
# coding: utf-8

# In[27]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime # for date time format
from matplotlib.ticker import FuncFormatter # to get y axis actual data instead of scientific number for plotting


# In[28]:


# read the dataset
df1 = pd.read_csv(r'D:\dataset\covid 19 dataset\CovidDeaths.csv')
df2 = pd.read_csv(r'D:\dataset\covid 19 dataset\CovidVaccinations.csv')


# In[29]:


# check 1st 5 row of data for dataframe1
df1.head(5)


# In[30]:


# check 1st 5 row of data for dataframe2
df2.head(5)


# In[31]:


# check the shape i.e rows and columns of both dataframe
df1.shape, df2.shape


# In[32]:


df1.info()


# In[33]:


df2.info()


# In[34]:


# use outerjoin to merge 2 dataframes
df= pd.merge(df1, df2, on=['iso_code','continent', 'location', 'date'], how='outer')


# In[35]:


# check the merged dataframe columns
df.columns


# In[36]:


# check for null values
print(df.isnull().sum())


# In[37]:


# drop the following columns
df.drop(['new_cases', 'new_deaths', 'new_deaths_smoothed', 'total_cases_per_million',
       'new_cases_per_million', 'new_cases_smoothed_per_million',
       'total_deaths_per_million', 'new_deaths_per_million',
       'new_deaths_smoothed_per_million', 'icu_patients',
       'icu_patients_per_million', 'hosp_patients',
       'hosp_patients_per_million', 'weekly_icu_admissions',
       'weekly_icu_admissions_per_million', 'weekly_hosp_admissions',
       'weekly_hosp_admissions_per_million', 'Unnamed: 26', 'Unnamed: 27','new_tests', 'total_tests_per_thousand',
       'new_tests_per_thousand', 'new_tests_smoothed',
       'new_tests_smoothed_per_thousand', 'positive_rate', 'tests_per_case',
       'tests_units', 'people_vaccinated', 'total_boosters', 'new_vaccinations',
       'new_vaccinations_smoothed', 'total_vaccinations_per_hundred',
       'people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred',
       'total_boosters_per_hundred', 'new_vaccinations_smoothed_per_million',
       'new_people_vaccinated_smoothed',
       'new_people_vaccinated_smoothed_per_hundred', 'stringency_index',
       'gdp_per_capita', 'extreme_poverty', 
       'diabetes_prevalence', 'female_smokers','handwashing_facilities', 'hospital_beds_per_thousand',
       'life_expectancy', 'human_development_index',
       'excess_mortality_cumulative_absolute', 'excess_mortality_cumulative',
       'excess_mortality', 'excess_mortality_cumulative_per_million','reproduction_rate','aged_65_older','median_age','aged_70_older','male_smokers'],
         axis = 1, inplace = True)


# In[38]:


# Filter rows where continent is NaN, and display along with location and iso_code columns
fil_continent_na= df1[df1['continent'].isna()][['location', 'iso_code','continent']].copy()
# Drop duplicates to get unique location and iso_code pairs
unique_loc = fil_continent_na.drop_duplicates()
unique_loc


# In[39]:


# Drop rows where continent is NaN
df= df.dropna(subset=['continent'])
print(df.isnull().sum())


# In[40]:


df.shape


# In[41]:


# Remove duplicates based on specified columns and keep the first occurrence
df = df.drop_duplicates(subset=['iso_code', 'continent', 'location', 'date'])


# In[42]:


df.shape


# In[43]:


# correlation matrix
num_df = df.select_dtypes(include=[np.number])
corr_matrix = num_df.corr()
corr_matrix


# In[44]:


# heat map
sns.heatmap(corr_matrix,annot=True,linewidths=0.5)


# In[45]:


# Format date

# replace '/' string with '-' in date column
df.loc[:, 'date'] = df['date'].str.replace('/', '-')

# Format the 'date' column to "day-month-year"
df.loc[:, 'date'] = pd.to_datetime(df['date'], format='%m-%d-%Y')


# In[46]:


# Sort the DataFrame by a relevant column (e.g., date) if needed
df = df.sort_values(by='date')


# In[47]:


# display top 5 cpountries by total Cases

# Aggregate total cases by country and Continent
tc_location = df.groupby(['location', 'continent'])['total_cases'].sum().reset_index()

# Sort the countries by total cases in descending order
sorted_tc = tc_location.sort_values(by='total_cases', ascending=False). head(5)

# Plot the bar graph
plt.figure(figsize=(8, 6))
ax=sns.barplot(x='location', y='total_cases', hue='continent', data=sorted_tc, palette='dark')
plt.xlabel('Country')
plt.ylabel('Total Cases')
plt.title('Top 5 Countries by Total Cases')

# Format y-axis to display numbers without scientific notation
plt.gca().get_yaxis().set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

# Annotate each bar with the actual number
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height):,}', 
                xy=(p.get_x() + p.get_width() / 2, height), 
                xytext=(0, 5),  # Offset label position
                textcoords='offset points',
                ha='center', va='center')

# Set y-axis limit with a bit of extra space
max_total_cases = sorted_tc['total_cases'].max()
plt.ylim(0, max_total_cases * 1.2)


# In[48]:


#death ratio

df['death_ratio'] = (df['total_deaths']/df['total_cases'])*100


# Sort the DataFrame by 'total_cases' in descending order
sorted_df = df.sort_values(by='total_cases', ascending=False)

# Group by 'location' and aggregate
grouped_df = sorted_df.groupby('location').agg({
    'total_cases': 'sum',
    'total_deaths': 'sum',
    'death_ratio': 'mean',
    'population': 'first'  # Assuming population remains the same for each location
}).reset_index()

# Sort by 'total_cases' in descending order
result_table = grouped_df.sort_values(by='total_cases', ascending=False).reset_index(drop=True)

# Format columns to display in readable format
result_table['total_cases'] = result_table['total_cases'].apply(lambda x: '{:,}'.format(x))
result_table['total_deaths'] = result_table['total_deaths'].apply(lambda x: '{:,}'.format(x))
result_table['death_ratio'] = result_table['death_ratio'].apply(lambda x: '{:.2f}'.format(x))
result_table['population'] = result_table['population'].apply(lambda x: '{:,}'.format(x))

# Display the resulting table

result_table[['location', 'population','total_cases', 'total_deaths','death_ratio']].rename(columns={'death_ratio': 'avg_death_ratio'}) .head(10)


# In[49]:


# % of population v/s deaths with respect to continent

continent_deaths = df.groupby('continent')['total_deaths'].sum().reset_index()

# Plotting
plt.figure(figsize=(8, 6))
plt.pie(continent_deaths['total_deaths'], labels=continent_deaths['continent'], autopct='%1.2f%%', startangle=120)
plt.title('Continent-wise Distribution of Deaths')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[50]:


# Convert the date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract month and year
df['month_year'] = df['date'].dt.to_period('M')

# Group by month_year and aggregate
grouped_df = df.groupby('month_year').agg({
    'total_cases': 'sum',
    'total_deaths': 'sum',
    'total_vaccinations': 'sum'
}).reset_index()

# Plotting
plt.figure(figsize=(15, 7))

# Plot total_cases
plt.plot(grouped_df['month_year'].astype(str), grouped_df['total_cases'], marker='o', label='Total Cases')

# Plot total_deaths
plt.plot(grouped_df['month_year'].astype(str), grouped_df['total_deaths'], marker='o', label='Total Deaths')

# Plot total_vaccinations
plt.plot(grouped_df['month_year'].astype(str), grouped_df['total_vaccinations'], marker='o', label='Total Vaccinations')

plt.xlabel('Month-Year')
plt.ylabel('Count')
plt.title('Total Cases, Deaths, and Vaccinations by Month-Year')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

# Formatter function to display actual numbers
formatter = FuncFormatter(lambda x, _: f'{int(x):,}')
plt.gca().yaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.show()

grouped_df


# In[51]:


#total cases in india with respect to location

# Filter for India and sum the required columns
india_data = df[df['location'] == 'India']
total_cases_sum = india_data['total_cases'].sum()
total_population_sum = india_data['population'].sum()

# Create summary table
summary_table = pd.DataFrame({
    'location': ['India'],
    'total_population_sum': [total_population_sum],
    'total_cases_sum': [total_cases_sum]
    
})

# Display summary table
pd.options.display.float_format = '{:,.0f}'.format  # Format numbers with commas and no decimals
print(summary_table)


# In[52]:


# Filter for Asia continent and India location
india_data = df[df['location'] == 'India']

# Group by location and calculate the sum of total_cases and population
grouped_df = india_data.groupby('location').agg({
   'population': 'first',
    'total_tests':'sum',
    'total_vaccinations':'sum',
    'total_cases': 'sum'
    
}).reset_index()

# Display the result with normal numbers
pd.options.display.float_format = '{:,.0f}'.format  # Format numbers with commas and no decimals
print(grouped_df)

