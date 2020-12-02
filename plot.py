import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

%matplotlib inline

train_df = pd.read_csv('/content/drive/My Drive/airbnb/train_users_2.csv')
age_gender_df = pd.read_csv('/content/drive/My Drive/airbnb/age_gender_bkts.csv')
countries_df = pd.read_csv('/content/drive/My Drive/airbnb/countries.csv')
submission_df = pd.read_csv('/content/drive/My Drive/airbnb/sample_submission_NDF.csv')
sessions_df = pd.read_csv('/content/drive/My Drive/airbnb/sessions.csv')
test_df = pd.read_csv('/content/drive/My Drive/airbnb/test_users.csv')
def popul_dest():
    sns.set_style('whitegrid')
    plt.figure(figsize=(10,7))
    pop_stats = age_gender_df.groupby('country_destination')['population_in_thousands'].sum()
    sns.barplot(x=pop_stats.index, y=pop_stats)
def dest_plt():
    sns.set_style('whitegrid')
    plt.figure(figsize=(10,7))
    sns.barplot(x='country_destination', y='distance_km', data=countries_df)
def device_type_plt():
    plt.figure(figsize=(12,7))
    sns.countplot(y='device_type', data=sessions_df)
def age_gender_plt():
    plt.figure(figsize=(20,8))
    sns.barplot(x='age_bucket', y='population_in_thousands', hue='gender', data=age_gender_df, ci=None)
def aff_prov_plt():
    #affilate provider plt
    plt.figure(figsize=(18,4))
    sns.countplot(train_df['affiliate_provider'])

def signup_app_plt():
    plt.figure(figsize=(18,4))
    sns.countplot(train_df['signup_app'])
def signup_method():
    plt.figure(figsize=(18,4))
    sns.countplot(train_df['signup_method'])
def dest_precentage():
    plt.figure(figsize=(10,5))
    country = train_df['country_destination'].value_counts() / train_df.shape[0] * 100
    country.plot(kind='bar',color='#FD5C64', rot=0)
    plt.xlabel('Destination Country')
    plt.ylabel('Percentage')
    sns.despine()

def main():
    popul_dest()
    est_plt()
    device_type_plt()
    age_gender_plt()
    aff_prov_plt()
    signup_app_plt()
    signup_method()
    dest_precentage()


main()