#!/usr/bin/env python
# coding: utf-8

# Training the KNN Model:

# 1) Apply feature scaling or normalization to ensure that all features have a similar scale.
# 2) Create an instance of the KNN model from a machine learning library like scikit-learn.
# 3) Fit the KNN model to the training data.

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# Assuming you have preprocessed training features and labels
train_features = ...
train_labels = ...

# Apply feature scaling
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(train_features)

# Create an instance of the KNN model
knn_model = KNeighborsRegressor(n_neighbors=5)

# Fit the KNN model to the scaled training data
knn_model.fit(scaled_train_features, train_labels)


# Predicting Hotel Rankings:

# 1) For each test instance (user), use the trained KNN model to find the k nearest neighbors in the training set based on their feature similarities.
# 2) Aggregate the rankings or preferences of the k nearest neighbors to predict the rankings for the test instance.
# 3) Optionally, you can use a weighted approach where closer neighbors have a higher influence on the predictions.

# In[ ]:


# Assuming you have preprocessed test features
test_features = ...

# Apply feature scaling to the test features using the same scaler instance
scaled_test_features = scaler.transform(test_features)

# Predict the rankings for the test instances
predictions = knn_model.predict(scaled_test_features)

# Optionally, consider a weighted approach
weighted_predictions = []
for indices, distances in knn_model.kneighbors(scaled_test_features):
    # Calculate weights based on inverse distances
    weights = 1 / distances

    # Apply weights to the rankings
    weighted_rankings = weights * train_labels[indices]

    # Aggregate the weighted rankings
    weighted_predictions.append(np.mean(weighted_rankings, axis=1))

# Final predictions as the mean of weighted predictions
final_predictions = np.mean(weighted_predictions, axis=0)


# Evaluating the Model:

# 1) Evaluate the performance of the KNN model by comparing the predicted hotel rankings with the actual rankings in the test set. You can use evaluation metrics such as precision, recall, or mean average precision to measure the model's effectiveness in maximizing purchases.

# In[ ]:


from sklearn.metrics import precision_score, recall_score, average_precision_score

# Assuming you have the true rankings in the test set
true_rankings = ...

# Evaluate precision
precision = precision_score(true_rankings, final_predictions)

# Evaluate recall
recall = recall_score(true_rankings, final_predictions)

# Evaluate mean average precision
average_precision = average_precision_score(true_rankings, final_predictions)

# Print the evaluation metrics
print("Precision:", precision)
print("Recall:", recall)
print("Mean Average Precision:", average_precision)


# Hyperparameter Tuning:

# 1) Experiment with different values of k (the number of neighbors) and other hyperparameters of the KNN algorithm to optimize the model's performance. You can use techniques like cross-validation or grid search to find the best hyperparameter values.

# In[ ]:


from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}

# Create an instance of the KNN model
knn_model = KNeighborsRegressor()

# Perform grid search with cross-validation
grid_search = GridSearchCV(knn_model, param_grid, cv=5)
grid_search.fit(scaled_train_features, train_labels)

# Get the best hyperparameters
best_n_neighbors = grid_search.best_params_['n_neighbors']
best_weights = grid_search.best_params_['weights']

# Create the optimized KNN model
optimized_knn_model = KNeighborsRegressor(n_neighbors=best_n_neighbors, weights=best_weights)
optimized_knn_model.fit(scaled_train_features, train_labels)


# Deployment and Testing:

# 1) Once you are satisfied with the model's performance, you can deploy it to make personalized hotel recommendations for new users. You can provide a list of top-ranked hotels based on their preferences and encourage them to make purchases.

# In[ ]:


# Deploy the optimized model for personalized hotel recommendations
def make_recommendations(user_features):
    scaled_user_features = scaler.transform(user_features)
    predictions = optimized_knn_model.predict(scaled_user_features)
    # Provide a list of top-ranked hotels based on predictions
    top_hotels = get_top_hotels(predictions)
    return top_hotels

