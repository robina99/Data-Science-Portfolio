# Import necessary libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import cluster
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import scale
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, completeness_score, homogeneity_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load survey dataset
survey_df = pd.read_csv('survey.csv', low_memory=False)

# Display the dataset's information
survey_df.info()

# Display the first 10 rows of the dataset
survey_df.head(10)

#Drop rows for survey that are not for tech companies
survey_df = survey_df[survey_df['tech_company'] != 'No']

# Display the dataset's information
survey_df.info()

# Check which values are in the tech_company column after dropping the ones that are not
print(survey_df['tech_company'].unique())

#Drop columns not required
survey = survey_df.drop(['Timestamp','comments', 'state', 'no_employees'], axis=1)

#Print missing values
missing_values = survey.isnull().sum()
print(missing_values)

# Fill the missing values with the mode
survey['self_employed'].fillna(survey['self_employed'].mode()[0], inplace=True)

#Print missing values
missing_values = survey.isnull().sum()
print(missing_values)

# Create a new category for unknown values in 'work_interfere'
survey['work_interfere'].fillna('Unknown', inplace=True)


#Print missing values
missing_values = survey.isnull().sum()
print(missing_values)

# Display the dataset's first 10 rows
survey.head(10)

# Display the statistics 
survey.describe()

# Display the datatype of the age column
print(survey['Age'].dtype)


# Display the values that exist in the age column
print(survey['Age'].unique())


# Clean the age column to replace the rows with less than 18 and more than 65 with the median of the age column
survey.loc[survey.Age<18, ["Age"]] = survey["Age"].median()
survey.loc[survey.Age>65, ["Age"]] = survey["Age"].median()
# Display the values that exist in the age column
survey["Age"].unique()

# Display the distribution of Ages
plt.figure(figsize=(6,4))
sns.histplot(survey['Age'], bins=5, kde=True, stat='percent')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Percentage')
plt.show()

# Display the statistics after cleaning the age column
survey.describe()

# Display the values that exist in the gender column
print(survey['Gender'].unique())

gender_mapping = {
    'Cis Male': 'Male',
    'male': 'Male',
    'M': 'Male',
    'm': 'Male',
    'Man': 'Male',
    'Trans Male': 'Others',
    'msle': 'Male',
    'maile': 'Male',
    'Mal': 'Male',
    'Make': 'Male',
    'Cis Man': 'Male',
    'Male (CIS)': 'Male',
    'Genderqueer': 'Others',
    'Male-ish': 'Others',
    'non-binary': 'Others',
    'Cis Female': 'Female',
    'female': 'Female',
    'F': 'Female',
    'f': 'Female',
    'Woman': 'Female',
    'woman': 'Female',
    'Trans-female': 'Others',
    'Femake': 'Female',
    'femail': 'Female',
    'Female ': 'Female',
    'cis-female/femme': 'Female',
    'Genderqueer': 'Others',
    'fluid': 'Others',
    'Agender': 'Others',
    'something kinda male?': 'Others',
    'queer/she/they': 'Others',
    'Nah': 'Others', 
    'All': 'Others', 
    'Enby': 'Others',
    'Androgyne': 'Others',
    'Guy (-ish) ^_^': 'Others', 
    'male leaning androgynous': 'Others', 
    'Trans woman': 'Others',
    'Neuter': 'Others', 
    'Female (trans)': 'Others', 
    'queer': 'Others',
    'Female (cis)': 'Female', 
    'Mail': 'Male', 
    'Male ': 'Male',
    'cis male': 'Male', 
    'A little about you': 'Others', 
    'Malr': 'Male', 
    'p': 'Others',
    'ostensibly male, unsure what that really means': 'Others',
    
}


# Standardize gender values
survey['Gender'] = survey['Gender'].replace(gender_mapping)

# Check unique values after standardization
print(survey['Gender'].unique())

# Calculate gender distribution and plot a pie chart
gender_counts = survey['Gender'].value_counts()

plt.figure(figsize=(6, 4))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2'))
plt.title('Gender Distribution')
plt.show()


#Bar chart showing distribution of participant countries
survey_location = survey['Country'].value_counts()
survey_location.plot(kind='bar', color='pink')
plt.figure(figsize=(2, 4))
plt.show()

# Bar chart showing the percentage distribution of remote work participants
survey_work = survey['remote_work'].value_counts(normalize=True) * 100 

plt.figure(figsize=(6, 4))
survey_work.plot(kind='bar', color='purple')
plt.title('Remote Work Distribution')
plt.ylabel('Percentage (%)')
plt.xlabel('Remote Work')
plt.show()


# Bar chart showing the percentage distribution of mental health consequence
survey_mental = survey['mental_health_consequence'].value_counts(normalize=True) * 100 

plt.figure(figsize=(6, 4))
survey_mental.plot(kind='bar', color='purple')
plt.title('Mental Health Consequence Distribution')
plt.ylabel('Percentage (%)')
plt.xlabel('Mental Health Consequence')
plt.show()

# Create a new column for age groups
age_bins = [0, 19, 29, 39, 49, 59, 69]  
age_labels = ['Under 20', '20-29', '30-39', '40-49', '50-59', '60-69']


survey['age_group'] = pd.cut(survey['Age'], bins=age_bins, labels=age_labels, right=False)


print(survey['age_group'].value_counts())


# Cross-tabulation of Age vs Mental Health Consequence
age_stigma_crosstab_percentage = pd.crosstab(
    survey['age_group'], survey['mental_health_consequence'], normalize='index') * 100

# Plotting Bar Plot with Percentages
plt.figure(figsize=(8, 6))
age_stigma_crosstab_percentage.plot(kind='bar', stacked=True, colormap="viridis")
plt.title('Age Group vs Mental Health Consequence')
plt.xlabel('Age Group')
plt.ylabel('Percentage')
plt.legend(title='Mental Health Consequences', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Cross-tabulation of gender vs Mental Health Consequence
gender_stigma_crosstab = pd.crosstab(survey['Gender'], survey['mental_health_consequence'], normalize='index') * 100

# Plotting Bar Plot
plt.figure(figsize=(6,4))
gender_stigma_crosstab.plot(kind='bar',  stacked=True, colormap="viridis")
plt.title('Gender vs Mental Health Consequence')
plt.xlabel('Gender')
plt.ylabel('Percentage')
plt.legend(title='Mental Health Consequence', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cross-tabulation of remote work vs Mental Health Consequence
remote_stigma_crosstab = pd.crosstab(survey['remote_work'], survey['mental_health_consequence'], normalize='index') * 100

# Plotting Bar Plot
plt.figure(figsize=(6,4))
remote_stigma_crosstab.plot(kind='bar',  stacked=True, colormap="viridis")
plt.title('Remote Work vs Mental Health Consequence')
plt.xlabel('Remote Work')
plt.ylabel('Percentage')
plt.legend(title='Mental Health Consequence', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Display the values that exist in the country column
print(survey['Country'].unique())

#map each country to its respective continent
country_to_continent = {
    'United States': 'North America', 'Canada': 'North America', 'Mexico': 'North America',
    'United Kingdom': 'Europe', 'Bulgaria': 'Europe', 'France': 'Europe', 'Portugal': 'Europe',
    'Netherlands': 'Europe', 'Switzerland': 'Europe', 'Poland': 'Europe', 'Germany': 'Europe',
    'Russia': 'Europe', 'Slovenia': 'Europe', 'Austria': 'Europe', 'Ireland': 'Europe',
    'Italy': 'Europe', 'Sweden': 'Europe', 'Latvia': 'Europe', 'Spain': 'Europe', 'Finland': 'Europe',
    'Uruguay': 'South America', 'Bosnia and Herzegovina': 'Europe', 'Hungary': 'Europe',
    'Croatia': 'Europe', 'Belgium': 'Europe', 'Norway': 'Europe', 'Greece': 'Europe', 'Moldova': 'Europe',
    'Costa Rica': 'North America', 'Brazil': 'South America', 'Colombia': 'South America',
    'Bahamas, The': 'North America', 'Australia': 'Oceania', 'New Zealand': 'Oceania',
    'Israel': 'Asia', 'India': 'Asia', 'Singapore': 'Asia', 'Japan': 'Asia', 'China': 'Asia',
    'Philippines': 'Asia', 'Thailand': 'Asia', 'South Africa': 'Africa', 'Zimbabwe': 'Africa',
    'Nigeria': 'Africa', 'Denmark': 'Europe', 'Czech Republic': 'Europe'
}

# Map countries to continents in a new 'Continent' column
survey['Continent'] = survey['Country'].map(country_to_continent)

#Bar chart showing distribution of participant continents
survey_continent = survey['Continent'].value_counts(normalize=True) * 100 
survey_continent.plot(kind='bar', color='lightcoral')
plt.ylabel('Percentage')
plt.title('Continent Distribution')
plt.figure(figsize=(2, 4))
plt.show()

# Cross-tabulation of continent vs Mental Health Consequence
country_stigma_crosstab = pd.crosstab(survey['Continent'], survey['mental_health_consequence'], normalize='index') * 100

# Plotting Bar Plot
plt.figure(figsize=(6,4))
country_stigma_crosstab.plot(kind='bar',  stacked=True, colormap="viridis")
plt.title('Continent vs Mental Health Consequence')
plt.xlabel('Continent')
plt.ylabel('Percentage')
plt.legend(title='Mental Health Consequence', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cross-tabulation of age group vs coworkers
agew_stigma_crosstab = pd.crosstab(survey['age_group'], survey['coworkers'], normalize='index') * 100

# Plotting Bar Plot
plt.figure(figsize=(6,4))
agew_stigma_crosstab.plot(kind='bar', stacked=True, colormap="plasma")
plt.title('Age Group vs Coworkers')
plt.xlabel('Age Group')
plt.ylabel('Percentage')
plt.legend(title='Coworkers', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cross-tabulation of gender vs coworkers
genderw_stigma_crosstab = pd.crosstab(survey['Gender'], survey['coworkers'], normalize='index') * 100

# Plotting Bar Plot
plt.figure(figsize=(6,4))
genderw_stigma_crosstab.plot(kind='bar', stacked=True, colormap="plasma")
plt.title('Gender vs Coworkers')
plt.xlabel('Gender')
plt.ylabel('Count of Responses')
plt.legend(title='Coworkers', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cross-tabulation of continent vs coworkers
countryw_stigma_crosstab = pd.crosstab(survey['Continent'], survey['coworkers'], normalize='index') * 100

# Plotting Bar Plot
plt.figure(figsize=(6,4))
countryw_stigma_crosstab.plot(kind='bar',  stacked=True, colormap="plasma")
plt.title('Continent vs Coworkers')
plt.xlabel('Continent')
plt.ylabel('Percentage')
plt.legend(title='Coworkers', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cross-tabulation of remote work vs coworkers
remotew_stigma_crosstab = pd.crosstab(survey['remote_work'], survey['coworkers'], normalize='index') * 100

# Plotting Bar Plot
plt.figure(figsize=(6,4))
remotew_stigma_crosstab.plot(kind='bar',  stacked=True, colormap="plasma")
plt.title('Remote Work vs Coworkers')
plt.xlabel('Remote Work')
plt.ylabel('Percentage')
plt.legend(title='Coworkers', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cross-tabulation of gender vs supervisor
genders_stigma_crosstab = pd.crosstab(survey['Gender'], survey['supervisor'], normalize='index') * 100

# Plotting Bar Plot
plt.figure(figsize=(6,4))
genders_stigma_crosstab.plot(kind='bar', stacked=True)
plt.title('Gender vs Supervisor')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.legend(title='Supervisor', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cross-tabulation of age group vs supervisor
ages_stigma_crosstab = pd.crosstab(survey['age_group'], survey['supervisor'], normalize='index') * 100

# Plotting Bar Plot
plt.figure(figsize=(6,4))
ages_stigma_crosstab.plot(kind='bar', stacked=True)
plt.title('Age Group vs Supervisor')
plt.xlabel('Age Group')
plt.ylabel('Percentage')
plt.legend(title='Supervisor', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cross-tabulation of continent vs supervisor
countrys_stigma_crosstab = pd.crosstab(survey['Continent'], survey['supervisor'], normalize='index') * 100

# Plotting Bar Plot
plt.figure(figsize=(6,4))
countrys_stigma_crosstab.plot(kind='bar',  stacked=True)
plt.title('Continent vs Supervisor')
plt.xlabel('Continent')
plt.ylabel('Percentage')
plt.legend(title='Supervisor', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cross-tabulation of remote work vs supervisor
remotes_stigma_crosstab = pd.crosstab(survey['remote_work'], survey['supervisor'], normalize='index') * 100

# Plotting Bar Plot
plt.figure(figsize=(6,4))
remotes_stigma_crosstab.plot(kind='bar',  stacked=True)
plt.title('Remote Work vs Supervisors')
plt.xlabel('Remote Work')
plt.ylabel('Percentage')
plt.legend(title='Supervisor', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Supervised Analysis 
label_encoder = LabelEncoder()
survey_target = survey['mental_health_consequence']

encoded_target = label_encoder.fit_transform(survey_target)

# Using only demographic columns
survey_feat = survey[['Gender', 'age_group', 'Continent', 'remote_work']].copy()

#survey_feat = survey.drop(['mental_health_consequence', 'Age', 'Country'], axis=1) # Remove the comments to run for other factors

for col in survey_feat.columns:
    le_encoder = LabelEncoder()
    survey_feat[col] = le_encoder.fit_transform(survey_feat[col])



X_train, X_test, Y_train, Y_test = model_selection.train_test_split(survey_feat, encoded_target, test_size=0.20, random_state=42)

# Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)

logistic_model.fit(X_train, Y_train)

predicted = logistic_model.predict(X_test)

accuracy = accuracy_score(Y_test, predicted)

print(f"Logistic Regression Accuracy: {accuracy:.2f}")

confusionMatrix = metrics.confusion_matrix(Y_test, predicted)
print(confusionMatrix)

ax = sns.heatmap(confusionMatrix / np.sum(confusionMatrix), annot=True, fmt='.1%', cmap='Blues')
plt.title('Confusion Matrix', fontsize = 20) # title with fontsize 20
plt.xlabel('Predicted values', fontsize = 15) # x-axis label with fontsize 15
plt.ylabel('True values', fontsize = 15) # y-axis label with fontsize 15
plt.show()

# Print Classification Report  # 0 For 'Maybe', # 1 For 'No', # 2 For 'Yes'
print(metrics.classification_report(Y_test, predicted))

correlation_data = survey_feat.copy()
correlation_data['mental_health_consequence'] = encoded_target
correlation_matrix = correlation_data.corr()

print(correlation_matrix)

plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Unsupervised Analysis 
label_encoder = LabelEncoder()

# Using only demographic columns
survey_feat_unsup = survey[['Gender', 'age_group', 'Continent', 'remote_work', 'coworkers', 'supervisor']].copy()


for col in survey_feat_unsup.columns:
    le_encoder = LabelEncoder()
    survey_feat_unsup[col] = le_encoder.fit_transform(survey_feat_unsup[col])

# Scale the data
scaler_unsup = StandardScaler()

survey_feat_unsup = scaler_unsup.fit_transform(survey_feat_unsup)

n_samples, n_features = survey_feat_unsup.shape
print("number of rows:", n_samples)
print("number of features:", n_features)

n_digits = len(np.unique(encoded_target))

print("number of different values for the target:", n_digits)

#KMeans clustering
kmeans = cluster.KMeans(n_clusters=n_digits)

kmeans.fit(survey_feat_unsup)
print(kmeans.get_params())

# silhouette score
print("Silhouette Coefficient:", metrics.silhouette_score(survey_feat_unsup, kmeans.labels_))

# completeness and the homogeneity scores
print("Completeness score", metrics.completeness_score(encoded_target, kmeans.labels_))
print("Homogeneity score", metrics.homogeneity_score(encoded_target, kmeans.labels_))

inertias = []

for i in range(1,21):
    kmeans = cluster.KMeans(n_clusters=i, n_init=10)
    kmeans.fit(survey_feat_unsup)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,21), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

cluster_range = range(2, 30)
silhouette_scores = []
completeness_scores = []
homogeneity_scores = []

# Iterate over each cluster count
for k in cluster_range:
    kmeans = cluster.KMeans(n_clusters=k, n_init=10)
    kmeans.fit(survey_feat_unsup)
    
    # Calculate metrics for the current number of clusters
    silhouette_avg = silhouette_score(survey_feat_unsup, kmeans.labels_)
    completeness = completeness_score(encoded_target, kmeans.labels_)
    homogeneity = homogeneity_score(encoded_target, kmeans.labels_)
    
    silhouette_scores.append(silhouette_avg)
    completeness_scores.append(completeness)
    homogeneity_scores.append(homogeneity)

# Plot all three scores on a single graph
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores, label='Silhouette Score', color='blue')
plt.plot(cluster_range, completeness_scores, label='Completeness Score', color='green')
plt.plot(cluster_range, homogeneity_scores, label='Homogeneity Score', color='red')

# Add labels and title
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Silhouette, Completeness, and Homogeneity Scores vs Number of Clusters')
plt.legend()
plt.grid(True)
plt.show()
