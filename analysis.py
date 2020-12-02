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

#read dataset
train_df = pd.read_csv('/content/drive/My Drive/airbnb/train_users_2.csv')
train_df = pd.read_csv('/content/drive/My Drive/airbnb/train_users_2.csv')
age_gender_df = pd.read_csv('/content/drive/My Drive/airbnb/age_gender_bkts.csv')
countries_df = pd.read_csv('/content/drive/My Drive/airbnb/countries.csv')
submission_df = pd.read_csv('/content/drive/My Drive/airbnb/sample_submission_NDF.csv')
sessions_df = pd.read_csv('/content/drive/My Drive/airbnb/sessions.csv')
test_df = pd.read_csv('/content/drive/My Drive/airbnb/test_users.csv')


# check each df has null attribtes
print('train df:',train_df.isnull().values.any(),'\n age_gender:',age_gender_df.isnull().values.any(),'\n countries:',countries_df.isnull().values.any()
,'\n submission df:',submission_df.isnull().values.any(),'\n session:',sessions_df.isnull().values.any(),'\n test:',test_df.isnull().values.any())

#check null for each file and take look to attributes in every file
#train file
def attibutes_value_counts():
    print('affiliate_provider',train_df['affiliate_provider'].value_counts())
    print('first_device_type',train_df['first_device_type'].value_counts())
    print('first_browser',train_df['first_browser'].value_counts())
    print('signup_method',train_df['signup_method'].value_counts())
    print('language',train_df['language'].value_counts())
    print('affiliate_channel',train_df['affiliate_channel'].value_counts())
    print('first_affiliate_tracked',train_df['first_affiliate_tracked'].value_counts())
    
def chi_sqr_test_affiliate_provider():
    #affilate provider vs country destination

    df_inf = train_df[(train_df['country_destination'] != 'NDF') & (train_df['country_destination'] != 'other') &(train_df['affiliate_provider'].notnull())]
    df_inf = df_inf[['id', 'affiliate_provider', 'country_destination']]
    table = df_inf.pivot_table('id', ['affiliate_provider'], 'country_destination', aggfunc='count').reset_index()
    table = table.set_index('affiliate_provider')
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return chi2,p


def chi_sqr_test_first_device_type():
    #first_device_type and country destination relation
    df_info = train_df[(train_df['country_destination'] != 'NDF') & (train_df['country_destination'] != 'other') &(train_df['first_device_type'].notnull())]
    df_info = df_info[['id', 'first_device_type', 'country_destination']]   
    table = df_info.pivot_table('id', ['first_device_type'], 'country_destination', aggfunc='count').reset_index()
    table = table.set_index('first_device_type')
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return chi2,p

def chi_sqr_test_first_browser():
    #first_browser and country destination relation
    df_info = train_df[(train_df['country_destination'] != 'NDF') & (train_df['country_destination'] != 'other') &(train_df['first_browser'].notnull())]
    df_info = df_info[['id', 'first_browser', 'country_destination']]
    table = df_info.pivot_table('id', ['first_browser'], 'country_destination', aggfunc='count').reset_index()
    table = table.set_index('first_browser')
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return chi2,p

def chi_sqr_test_gender():
    #gender and country destination relation
    df_inf = train_df[(train_df['country_destination'] != 'NDF') & (train_df['country_destination'] != 'other') & (train_df['gender'] != 'OTHER') & (train_df['gender'].notnull())]
    df_inf = df_inf[['id', 'gender', 'country_destination']]
    table = df_inf.pivot_table('id', ['gender'], 'country_destination', aggfunc='count').reset_index()
    table = table.set_index('gender')
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return chi2,p

def chi_sqr_test_affiliate_channel():
    #affiliate_channel vs country destination
    df_inf = train_df[(train_df['country_destination'] != 'NDF') & (train_df['country_destination'] != 'other') &  (train_df['affiliate_channel'].notnull())]
    df_inf = df_inf[['id', 'affiliate_channel', 'country_destination']]
    table = df_inf.pivot_table('id', ['affiliate_channel'], 'country_destination', aggfunc='count').reset_index()
    table = table.set_index('affiliate_channel')
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return chi2,p

def chi_sqr_test_first_affiliate_tracked():
    #first_affiliate_tracked and country destination relation
    df_inf = train_df[(train_df['country_destination'] != 'NDF') & (train_df['country_destination'] != 'other') &(train_df['first_affiliate_tracked'].notnull())]
    df_inf = df_inf[['id', 'first_affiliate_tracked', 'country_destination']]
    table = df_inf.pivot_table('id', ['first_affiliate_tracked'], 'country_destination', aggfunc='count').reset_index()
    table = table.set_index('first_affiliate_tracked')
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return chi2,p



def train_df_feature_eng():
    #now age attribute hasn't outliers and NaN values 
    #handle missing values in gender attribute
    train_df['gender']=train_df['gender'].replace('-unknown-',np.nan)
    train_df['gender']=train_df['gender'].fillna('Unknown')
    #handle missing values in first browser attribute
    train_df['first_browser']=train_df['first_browser'].replace('-unknown-',np.nan)
    #replace all outliers in age attribute that have (120 year) by NAN 
    train_df['age'] = train_df['age'].apply(lambda x: np.nan if x > 120 else x)
    #handle missing values and merge marketing and local ops to other 
    train_df['first_affiliate_tracked']= train_df['first_affiliate_tracked'].fillna('Unknown')
    train_df['first_affiliate_tracked'] = train_df['first_affiliate_tracked'].apply(lambda x: 'untracked' if x == 'marketing' or x == 'local ops' else x)

    #hancle proveders and combine least impact to rest
    train_df['affiliate_provider'] = train_df['affiliate_provider'].apply(lambda x: 'rest' if x not in ['direct', 'google', 'other','craigslist','bing','facebook'] else x)

    #combine all desktops and tablet and phones and any other value considred as Unknown 
    train_df['first_device_type'] = train_df['first_device_type'].apply(lambda x: 'Desktop' if x.find('Desktop')!=-1 else x)
    train_df['first_device_type'] = train_df['first_device_type'].apply(lambda x: 'Tablet' if x.find('Tablet')!=-1 or x.find('iPad') != -1 else x)
    train_df['first_device_type'] = train_df['first_device_type'].apply(lambda x: 'Phone' if x.find('Phone')!=-1 else x)
    train_df['first_device_type'] = train_df['first_device_type'].apply(lambda x: 'Unknown' if x not in ['Desktop','Tablet','Phone']  else x)
    #handling Unknown values and missing values and combine least impact values to other  
    train_df['first_browser'] = train_df['first_browser'].replace('-unknown-', np.nan)
    train_df['first_browser']= train_df['first_browser'].fillna('Unknown')
    train_df['first_browser'] = train_df['first_browser'].apply(lambda x: 'Moblie_safari' if x=='Mobile Safari' else x)
    train_df['first_browser'] = train_df['first_browser'].apply(lambda x: 'Other' if x not in ['Chrome','Safari','Firefox','IE','Mobile_safari','Unknown'] else x)
    #combine all language in foreign , engligh 
    train_df['language'] = train_df['language'].apply(lambda x: 'foreign' if x != 'en' else x)
    #date account hasn't relation with dest
    train_df = train_df.drop('date_first_booking', axis=1)
    #date first booking same
    train_df = train_df.drop('date_account_created', axis=1)
    #
    train_df['is_3'] = train_df['signup_flow'].apply(lambda x: 1 if x==3 else 0)
    #timestamp hasn't relation with country destination 
    train_df = train_df.drop('timestamp_first_active', axis=1)
    #ids handling
    train_df.set_index('id')

def session_df_cleaning():
    #handle session file 
    sessions_df['action'] = sessions_df['action'].apply(lambda x: np.nan if x =='-unknown-' else x)
    sessions_df['action']= sessions_df['action'].fillna('Unknown')

    sessions_df['action_type'] = sessions_df['action_type'].apply(lambda x: np.nan if x =='-unknown-' else x)

    sessions_df['action_type']= sessions_df['action_type'].fillna('Unknown')


    sessions_df['action_detail'] = sessions_df['action_detail'].apply(lambda x: np.nan if x =='-unknown-' else x)
    sessions_df['action_detail']= sessions_df['action_detail'].fillna('Unknown')

    sessions_df['device_type'] = sessions_df['device_type'].apply(lambda x: 'Desktop' if x.find('Desktop')!=-1 else x)
    sessions_df['device_type'] = sessions_df['device_type'].apply(lambda x: 'Tablet' if x.find('Tablet')!=-1 or x.find('iPad') != -1 else x)
    sessions_df['device_type'] = sessions_df['device_type'].apply(lambda x: 'Phone' if x.find('Phone')!=-1 else x)
    sessions_df['device_type'] = sessions_df['device_type'].apply(lambda x: 'Unknown' if x not in ['Desktop','Tablet','Phone'] else x)



    sessions_df[sessions_df['user_id'].isna()].head()

def age_gender_cleaning():
    #handle gender df (age bucket attribute)
    age_gender_df['age_bucket'] = age_gender_df['age_bucket'].apply(lambda x: '100-104' if x == '100+' else x)
    age_gender_df['age_bucket'] = age_gender_df['age_bucket'].apply(lambda x: (int(x.split('-')[0]) + int(x.split('-')[1]))/2)
    age_gender_df['gender'] = age_gender_df['gender'].apply(lambda x: 1 if x == 'male' else 0)
    age_gender_df= age_gender_df.drop('year', axis=1)

def main():
    session_df_cleaning()
    age_gender_cleaning()
    train_df_feature_eng()




main()