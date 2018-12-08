import pandas as pd
import decode as dc

dataset = pd.read_csv("true_car_listings.csv", header=0)
print("Inicial formato: ", dataset.shape)

dataset = dataset.drop_duplicates()
print("Removendo duplicados: ", dataset.shape)

dataset['Model'] = dataset['Model'].str.replace(',', '')
dataset['Model'] = dataset['Model'].str.capitalize()
dataset['City'] = dataset['City'].str.capitalize()
dataset['Make'] = dataset['Make'].str.capitalize()
dataset['Country_Make'] = dataset['Vin'].astype(str).str[0]

region = []
for i in range(dataset.shape[0]):
    region.append(dc.region(dataset.iloc[i]["Vin"]))
dataset["Region"] = region

country_digit_price_mean = dataset.groupby(['Country_Make'])["Price"].mean().reset_index(
    name='mean_price_country_digit')
country_digit_price_max = dataset.groupby(['Country_Make'])["Price"].max().reset_index(
    name='max_price_country_digit')
country_digit_price_min = dataset.groupby(['Country_Make'])["Price"].min().reset_index(
    name='min_price_country_digit')
country_digit_price_std = dataset.groupby(['Country_Make'])["Price"].std().reset_index(
    name='std_price_country_digit')

region_price_mean = dataset.groupby(['Region'])["Price"].mean().reset_index(name='mean_price_region')
region_price_max = dataset.groupby(['Region'])["Price"].max().reset_index(name='max_price_region')
region_price_min = dataset.groupby(['Region'])["Price"].min().reset_index(name='min_price_region')
region_price_std = dataset.groupby(['Region'])["Price"].std().reset_index(name='std_price_region')

sc_price_mean = dataset.groupby(['State', "City"])["Price"].mean().reset_index(name='mean_price_sc')
sc_price_max = dataset.groupby(['State', "City"])["Price"].max().reset_index(name='max_price_sc')
sc_price_min = dataset.groupby(['State', "City"])["Price"].min().reset_index(name='min_price_sc')
sc_price_std = dataset.groupby(['State', "City"])["Price"].std().reset_index(name='std_price_sc')

mm_price_mean = dataset.groupby(['Make', "Model"])["Price"].mean().reset_index(name='mean_price_mm')
mm_price_max = dataset.groupby(['Make', "Model"])["Price"].max().reset_index(name='max_price_mm')
mm_price_min = dataset.groupby(['Make', "Model"])["Price"].min().reset_index(name='min_price_mm')
mm_price_std = dataset.groupby(['Make', "Model"])["Price"].std().reset_index(name='std_price_mm')

dataset = pd.merge(dataset, country_digit_price_mean, on='Country_Make', how='left')
dataset = pd.merge(dataset, country_digit_price_mean, on='Country_Make', how='left')
dataset = pd.merge(dataset, country_digit_price_mean, on='Country_Make', how='left')
dataset = pd.merge(dataset, country_digit_price_mean, on='Country_Make', how='left')

dataset = pd.merge(dataset, region_price_mean, on='Region', how='left')
dataset = pd.merge(dataset, region_price_max, on='Region', how='left')
dataset = pd.merge(dataset, region_price_min, on='Region', how='left')
dataset = pd.merge(dataset, region_price_std, on='Region', how='left')

dataset = pd.merge(dataset, sc_price_mean, left_on=['State', 'City'], right_on=['State', 'City'], how='left')
dataset = pd.merge(dataset, sc_price_max, left_on=['State', 'City'], right_on=['State', 'City'], how='left')
dataset = pd.merge(dataset, sc_price_min, left_on=['State', 'City'], right_on=['State', 'City'], how='left')
dataset = pd.merge(dataset, sc_price_std, left_on=['State', 'City'], right_on=['State', 'City'], how='left')

dataset = pd.merge(dataset, mm_price_mean, left_on=['Make', 'Model'], right_on=['Make', 'Model'], how='left')
dataset = pd.merge(dataset, mm_price_max, left_on=['Make', 'Model'], right_on=['Make', 'Model'], how='left')
dataset = pd.merge(dataset, mm_price_min, left_on=['Make', 'Model'], right_on=['Make', 'Model'], how='left')
dataset = pd.merge(dataset, mm_price_std, left_on=['Make', 'Model'], right_on=['Make', 'Model'], how='left')


dataset = dataset.loc[:, dataset.columns != 'City']
dataset = dataset.loc[:, dataset.columns != 'State']
dataset = dataset.loc[:, dataset.columns != 'Make']
dataset = dataset.loc[:, dataset.columns != 'Model']
dataset = dataset.loc[:, dataset.columns != 'Country_Make']
dataset = dataset.loc[:, dataset.columns != 'Vin']
dataset.to_csv("true_car_listings_clean.csv", index=False)
