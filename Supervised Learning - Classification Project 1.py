#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Dataset: Binary Classification Problem

# In[3]:


# Preparing tools for this Project
# perform EDA and plotting data (Add Libraries)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Our graphs can appear inside of notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Import Models from Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, RocCurveDisplay


# In[4]:


heart_disease_df = pd.read_csv('heart-disease.csv')
heart_disease_df


# In[5]:


heart_disease_df.shape #rows and colums


# # Let's do Data Exploration Analysis or EDA
# 
# The goal here is to find more about the data and become subject matter expert on the dataset you're working on.
# 
# 1. What questions(s) are we trying to solve?
# 2. What kind of data do we have and how do we treat different types?
# 3. What's missing from the data and how do we deal with it?
# 4. where are the outliers and why should you care about them?
# 5. How can you add, change or remove features to get more out of your data?

# In[6]:


heart_disease_df.head()


# In[7]:


heart_disease_df.tail()


# In[8]:


heart_disease_df['target'].value_counts()


# In[9]:


heart_disease_df['target'].value_counts().plot(kind='bar', color=['blue', 'red'])


# In[10]:


heart_disease_df.info()


# In[11]:


heart_disease_df.isna().sum()


# In[12]:


heart_disease_df.describe()


# # Heart Disease Frequencey according to Sex

# In[13]:


heart_disease_df.sex.value_counts()


# #### Compare target column with sex column

# In[14]:


pd.crosstab(heart_disease_df.target, heart_disease_df.sex)


# In[15]:


#Create a plot of crosstab
pd.crosstab(heart_disease_df.target, heart_disease_df.sex).plot(kind='bar',
                                                               figsize=(10,6),
                                                               color=['red', 'blue'])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('0 = No Disease, 1 = Disease')
plt.ylabel('Amount')
plt.legend(['Female', 'Male']);
plt.xticks(rotation=0)


# ## Compare Age with Max Heart Rate achieved

# In[16]:


#Create another figure
plt.figure(figsize=(10,6))

#Scatter with positive examples
plt.scatter(heart_disease_df.age[heart_disease_df.target==1],
           heart_disease_df.thalach[heart_disease_df.target==1],
           c='salmon')

#Scatter with negative examples
plt.scatter(heart_disease_df.age[heart_disease_df.target==0],
           heart_disease_df.thalach[heart_disease_df.target==0],
           c='lightblue')

#Add some helpful info
plt.title('Heart Disease in function of Age and Max Heart Rate')
plt.xlabel('Age')
plt.ylabel('Max Heart Rate')
plt.legend(['Disease', 'No Disease']);


# In[17]:


# Checking the distribution of the age column with a histogram
heart_disease_df.age.plot.hist();


# ### Heart Disease Frequency per Chest Pain Type

# In[18]:


pd.crosstab(heart_disease_df.cp, heart_disease_df.target)


# In[19]:


# Make the crosstab more visual
pd.crosstab(heart_disease_df.cp, heart_disease_df.target).plot(kind='bar',
                                                              figsize=(10,6),
                                                              color=['red', 'blue'])
# Add communication
plt.title('Heart Disease Frequency Per Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.ylabel('Amount')
plt.legend(['No Disease', 'Disease'])
plt.xticks(rotation=0);


# In[20]:


heart_disease_df.head()


# In[21]:


#Make a correlation matrix
heart_disease_df.corr()


# In[22]:


#Let's turn this into a visualisation to make it more conveyable
corr_matrix = heart_disease_df.corr()
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
                annot=True,
                linewidths=0.5,
                fmt='.2f',
                cmap='YlGnBu');


# Now we've got our data split into training and test datsets, it's time to build a machine learning model.
# 
# We'll train the data on training set and use the patterns on test set.
# 
# We choose 3 models to work with our heart disease set:
# 1. Logistic Regression
# 2. K-Neighbors Classifier
# 3. Random Forest Classifier
# 

# # 5. Modelling

# In[23]:


heart_disease_df.head()


# In[24]:


#Split data into X and y
X = heart_disease_df.drop('target', axis=1)
y = heart_disease_df['target']


# In[25]:


X


# In[26]:


y


# In[27]:


# Split data train and test sets
np.random.seed(42)

#Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                  test_size=0.2)


# In[28]:


X_train


# In[29]:


y_train


# In[30]:


# Put models in a dictionary
models = {'Logistic Regression': LogisticRegression(),
         'KNN': KNeighborsClassifier(),
         'Random Forest': RandomForestClassifier()}

# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    '''
    Fits and evaluates given machine learning models.
    models: a dict of different SciKkit-Learn machine learning models
    X_train: training data (no labels)
    X_test: test data (no labels)
    y_train: traning labels
    y_test: testing labels
    '''
    
    # Set random seeds
    np.random.seed(23)
    # Make a dictionary to keep model scores
    model_scores = {}
    #Loop through models
    for name, model in models.items():
        #Fit the model to the data
        model.fit(X_train, y_train)
        #Evaluate the model and append it's score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores


# In[31]:


model_scores = fit_and_score(models=models, 
                             X_train=X_train, 
                             X_test=X_test, 
                             y_train=y_train, 
                             y_test=y_test)
print(model_scores)


# ### Model Comparison

# In[32]:


model_compare = pd.DataFrame(model_scores, index=['accuracy'])
model_compare.T.plot.bar();


# We have got our baseline model. But we can move to the next steps of the evaluation depending on our first based predictions.
# 
# Let's look at the following:
# * Hyperparameter Tuning
# * Feature Importance
# * Confusion Matrix
# * Cross-validation
# * Precision
# * Recall
# * F1 score
# * Classification Report
# * ROC curve
# * Area under the curve (AUC)
# 
# ### Hyperparameter tuning (by hand)

# In[33]:


# let's tune KNN

train_scores = []
test_scores = []

# Create a list of different values for n_neighbors
neighbors = range(1, 21)

# Setup KNN instance
knn = KNeighborsClassifier()

#Loop through differenct n_neighbors
for i in neighbors:
    knn.set_params(n_neighbors=i)
    
    # Fit the algorithm
    knn.fit(X_train, y_train)
    
    #Update the training scores list
    train_scores.append(knn.score(X_train, y_train))
    
    #Update the test scores list
    test_scores.append(knn.score(X_test, y_test))


# In[34]:


train_scores, test_scores


# In[35]:


plt.plot(neighbors, train_scores, label='Train score')
plt.plot(neighbors, test_scores, label='Test score')
plt.xticks(np.arange(1,21,1))
plt.xlabel('Number of neighbors')
plt.ylabel('Model score')
plt.legend()

print(f"Maximum KNN score on the training data: {max(train_scores)*100:.2f}%")
print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")


# ## Hyperparameter tuning with RandomizedSearchCV
# 
# We're going to tune:
# * Logistc Regression
# * Random Forest Classifier
# 
# ..... by RandomSearchedCV

# In[36]:


# Create a hyperparameter grid for LogisticRegression
log_reg_grid = {'C': np.logspace(-4,4,20),
               'solver': ['liblinear']}

#Create a hyperparamter grid for RandomForestClassifier
ran_for_grid = {'n_estimators': np.arange(10,1000,50),
               'max_depth': [None,3,5,10],
               'min_samples_split': np.arange(2,20,2),
               'min_samples_leaf': np.arange(1,20,2)}


# **Now we've got hyperparameter grids setup for each of our models, let's tune them using RandomizedSearchCV...

# In[37]:


# Tune LogisticRegression

np.random.seed(42)

#Setup random hyperparameter search for Logistic Regression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                               param_distributions=log_reg_grid,
                               cv=5,
                               n_iter=20,
                               verbose=True)

#Fit random hyperparameter search model for LogisticRegression
rs_log_reg.fit(X_train, y_train)


# In[38]:


rs_log_reg.best_params_


# In[39]:


rs_log_reg.score(X_test,y_test)


# Now we've tuned LogisticRegression(), let's do the same for RandomForestClassifier()....

# In[40]:


# Tune RandomForestClassifier

np.random.seed(42)

#Setup random hyperparameter search for RandomForest
rs_ran_for_ = RandomizedSearchCV(RandomForestClassifier(),
                               param_distributions=ran_for_grid,
                               cv=5,
                               n_iter=20,
                               verbose=True)

#Fit random hyperparameter search model for RandomForest
rs_ran_for_.fit(X_train, y_train)


# In[41]:


rs_ran_for_.best_params_


# In[42]:


rs_ran_for_.score(X_test,y_test)


# 1. By hand
# 2. RandomizedSearchCV
# 3. GridSearchCV

# ## Hyperparameter tuning with GridSearchCV
# 
# Since our LogisticRegression model provides the best scores so far, we'll try and improve them again using GridSearchCV....

# In[43]:


#Different hyperparameters for our LogisticRegression model
log_reg_grid = {'C': np.logspace(-4,4,30),
               'solver': ['liblinear']}

#Setup grid hyperparameter search for LogisticRegression
gs_log_reg = GridSearchCV(LogisticRegression(),
                         param_grid=log_reg_grid,
                         cv=5,
                         verbose=True)

# Fit grid hyperparameter search model
gs_log_reg.fit(X_train,y_train)


# In[44]:


gs_log_reg.best_params_


# In[45]:


gs_log_reg.score(X_test,y_test)


# In[46]:


model_scores


# In[47]:


# Create a hyperparamter grid for RandomForestClassifier
ran_for_grid = {'n_estimators': np.arange(10,1000,50),
               'max_depth': [None,3,5,10],
               'min_samples_split': np.arange(2,20,2),
               'min_samples_leaf': np.arange(1,20,2)}

# Setup grid hyperparameter search for RandomForestClassifier
gs_ran_for = GridSearchCV(RandomForestClassifier(),
                         param_grid=ran_for_grid,
                         cv=5,
                         verbose=True)
# Fit grid hyperparameter search model
#gs_ran_for.fit(X_train,y_train)


# ## Evaluting our tuned machine learning classifier, beyond accuracy
# 
# * ROC curve and AUC score
# * Confusion matrix
# * Classification report
# * Precision 
# * Recall
# * F1-score
# 
# ...and it would be great if cross-validation was used where possible.
# 
# To make comparisons and evaluate out trained model, first we need to make predictions.

# In[48]:


# Make predictions with tuned model
y_preds = gs_log_reg.predict(X_test)


# In[49]:


y_preds


# In[50]:


y_test


# In[51]:


# Plot ROC curve and calculate AUC metric
RocCurveDisplay.from_estimator(gs_log_reg, X_test, y_test);


# In[52]:


# Confusion Matrix
confusion_matrix(y_test, y_preds)


# In[53]:


# Import Seaborn
import seaborn as sns
sns.set(font_scale=1.5) # Increase font size
 
def plot_conf_mat(y_test, y_preds):
    """
    Plots a confusion matrix using Seaborn's heatmap().
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True, # Annotate the boxes
                     cbar=False)
    plt.xlabel("Predicted label") # predictions go on the x-axis
    plt.ylabel("True label") # true labels go on the y-axis 
    
plot_conf_mat(y_test, y_preds)


# In[54]:


print(classification_report(y_test,y_preds)) 


# ### Calculate evaluation metrics using cross-validation
# 
# We're going to calculate precision, recall and f1-score of our model using cross-calidation and to do so we'll be using `cross_val_score()`

# In[55]:


# Check best hyperparameters
gs_log_reg.best_params_


# In[57]:


# Creatse a new classifier with best parameters
clf = LogisticRegression(C=0.20433597178569418,
                         solver='liblinear')


# In[73]:


# Cross-validated accuracy

cv_acc = cross_val_score(clf,
                        X,
                        y,
                        scoring='accuracy')
cv_acc

cv_acc_mean = np.mean(cv_acc)
cv_acc_mean


# In[74]:


# Cross-validated precision

cv_precision = cross_val_score(clf,
                        X,
                        y,
                        scoring='precision')
cv_precision

cv_precision_mean = np.mean(cv_precision)
cv_precision_mean


# In[75]:


# Cross-validated recall

cv_recall = cross_val_score(clf,
                        X,
                        y,
                        scoring='recall')
cv_recall

cv_recall_mean = np.mean(cv_recall)
cv_recall_mean


# In[77]:


# Cross-validated f1-score

cv_f1_score = cross_val_score(clf,
                        X,
                        y,
                        scoring='f1')
cv_f1_score

cv_f1_score_mean = np.mean(cv_f1_score)
cv_f1_score_mean


# In[79]:


# Visualize cross-validated metrics
cv_metrics = pd.DataFrame({'Accuracy': cv_acc_mean,
                          'Precision': cv_precision_mean,
                          'Recall': cv_recall_mean,
                          'F1': cv_f1_score_mean},
                         index=[0])

cv_metrics.T.plot.bar(title='Cross-validatdd classfication metrics',
                     legend=False);


# ### Feature Importance
# 
# Feature importance is another way of asking, 'which features contributed most to the outcomes of the model and how did they contribute?'
# 
# Finding feature importance is different for each machine learning model. One way tof ind feature importance is to search for '(MODEL NAME) feature importance'.
# 
# Let's fond the feature importance for our LogisticRegression model...

# In[80]:


heart_disease_df.head()


# In[81]:


# Find an instance of LogisticRegression
clf = LogisticRegression(C=0.20433597178569418,
                         solver='liblinear')

clf.fit(X_train, y_train)


# In[82]:


clf.coef_


# In[84]:


# Match coef's of features to columns
features_dict = dict(zip(heart_disease_df, list(clf.coef_[0])))
features_dict


# In[87]:


#Visualize feature importance
features_df = pd.DataFrame(features_dict, index=[0])
features_df.T.plot.bar(title="Feature Importance", legend=False);


# In[88]:


pd.crosstab(heart_disease_df.target, heart_disease_df.sex)


# In[90]:


72/24, 114/93


# In[92]:


pd.crosstab(heart_disease_df.slope, heart_disease_df.target)


# ## Experimentation

# In[ ]:




