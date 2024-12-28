
def fetch_credit_dataset():
    if not os.path.exists("data/credit.zip"):
        r = requests.get("https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip")
        with open("data/credit.zip", "wb") as f:
            f.write(r.content)
        
        with zipfile.ZipFile("data/credit.zip", 'r') as zip_ref:
            zip_ref.extractall("data") 


    return pd.read_excel("data/default of credit card clients.xls")


raw_credit_data = fetch_credit_dataset()


raw_credit_data = raw_credit_data.set_axis(raw_credit_data.iloc[0, :], axis=1)
raw_credit_data.drop(labels=[0], axis=0, inplace=True)
raw_credit_data.drop(labels=['ID'], axis=1, inplace=True)
raw_credit_data.rename(columns={'default payment next month': 'y'}, inplace=True)
numeric_cols = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
for col in numeric_cols:
    raw_credit_data[col] = pd.to_numeric(raw_credit_data[col])


print(len(raw_credit_data))


#  Checking for missing values


missing_values = -1*(raw_credit_data.count()-len(raw_credit_data))
print(missing_values)


#  Checking for class imbalance
# High class imbalance with the negative class having majority samples


label_count = raw_credit_data.value_counts(['y'])
fig, ax = plt.subplots()
ax.bar([0, 1], label_count.to_numpy())
ax.set_xticks([0, 1], labels=label_count.index)
plt.show()
plt.close()


#  Splitting dataset


train_credit_data, test_credit_data = train_test_split(raw_credit_data, test_size=0.2)
val_credit_data, test_credit_data = train_test_split(test_credit_data, test_size=0.5)
print(train_credit_data.value_counts(['y']), val_credit_data.value_counts(['y']), test_credit_data.value_counts(['y']))


# train_credit_data = train_credit_data.drop(train_credit_data[train_credit_data['y'] == 0].sample(frac=0.67).index)
# print(train_credit_data.value_counts(['y']))


#  Training Logistic Classifier


class LogisticRegressionEstimator:
    def __init__(self, data, lr) -> None:
        self.lr = lr

        self.y = pd.to_numeric(data['y']).to_numpy()
        self.features = data.drop("y", axis=1)
        for col in self.features.columns:
            self.features[col] = pd.to_numeric(self.features[col])
        self.features = self.features.to_numpy()
        
        self.min_features, self.max_features = self.features.min(axis=0, keepdims=False), self.features.max(axis=0, keepdims=False)
        self.features = ((self.features-self.min_features)/(self.max_features-self.min_features))
        self.parameters =  np.random.rand(self.features.shape[1])
    
    def step(self):
        intermediate = np.matmul(self.features, self.parameters)
        intermediate = np.reciprocal(1 + np.exp(-1*intermediate))
        
        gradient = -1*(self.y-intermediate)

        gradient = np.matmul(self.features.T, gradient)
        self.parameters = self.parameters - self.lr*gradient

    def predict(self, x, threshold=0.5):
        if isinstance(x, pd.Series):
            if 'y' in x.index:
                x = x.drop('y')
            x = x.to_numpy()
        elif isinstance(x, pd.DataFrame):
            if 'y' in x.columns:
                x = x.drop('y', axis=1)
            x = x.to_numpy()

        x = (x - self.min_features)/(self.max_features - self.min_features)
        val =  1/(1 + np.exp((-1*np.matmul(x, self.parameters))))
        if val >= threshold:
            return 1
        return 0


def training_loop(model, epochs, train_data, val_data, epoch_delta=500):
    train_accs, val_accs = [], []
    for epoch in range(epochs):
        model.step()

        if epoch%epoch_delta == 0:
            train_predictions = []
            for _, row in train_data.iterrows():
                train_predictions.append(model.predict(row))
            val_predictions = []
            for _, row in val_data.iterrows():
                val_predictions.append(model.predict(row))
                
            train_acc, _, _ = confusion_matrix(train_predictions, train_data['y'].to_numpy(), stat_labels=[1], display_matrix=False)
            val_acc, _, _ = confusion_matrix(val_predictions, val_data['y'].to_numpy(), stat_labels=[1], display_matrix=False)

            train_accs.append(train_acc*100)
            val_accs.append(val_acc*100)
    
    fig, ax = plt.subplots()
    ax.plot(range(len(train_accs)), train_accs, color="blue")
    ax.plot(range(len(train_accs)), val_accs, color="orange")
    ax.set_xlabel("Epochs (*%s)"%epoch_delta)
    ax.set_ylabel("%")
    ax.legend(["Train Accuracies", "Validation Accuracies"])
    # plt.show()
    # plt.close()



lre = LogisticRegressionEstimator(train_credit_data, 0.00001)
training_loop(lre, 10000, train_credit_data, val_credit_data, epoch_delta=1000)


predictions = []
for _, row in test_credit_data.iterrows():
    predictions.append(lre.predict(row))


lr_accuracy, lr_vals, lr_auc = confusion_matrix(predictions, test_credit_data["y"].to_numpy(), [1])


#  KNN Classifier


class KNNClassifier:
    def __init__(self, data, k) -> None:
        self.k = k
        self.y = data['y'].to_numpy()
        self.features = data.drop("y", axis=1)
        for col in self.features.columns:
            self.features[col] = pd.to_numeric(self.features[col])
        self.features = self.features.to_numpy()
        
        self.min_features, self.max_features = self.features.min(axis=0, keepdims=False), self.features.max(axis=0, keepdims=False)
        self.features = ((self.features-self.min_features)/(self.max_features-self.min_features))
        self.parameters =  np.random.rand(self.features.shape[1])
    
    def predict(self, x):
        if isinstance(x, pd.Series):
            if 'y' in x.index:
                x = x.drop('y')
            x = x.to_numpy()
        elif isinstance(x, pd.DataFrame):
            if 'y' in x.columns:
                x = x.drop('y', axis=1)
            x = x.to_numpy()

        x = (x - self.min_features)/(self.max_features - self.min_features)

        nearest = []
        for i in range(self.features.shape[0]):
            dist = np.linalg.norm(self.features[i] - x)

            j = len(nearest)-1
            nearest.append((0,0))
            while j >= 0 and nearest[j][0] > dist:
                nearest[j+1] = nearest[j]
                j -= 1
            nearest[j+1] = (dist, i)
            if len(nearest) > self.k:
                nearest.pop()
        
        dic = {}
        max_val = 0
        prediction = None
        for dist, index in nearest:
            if self.y[index] not in dic.keys():
                dic[self.y[index]] = 0
            
            dic[self.y[index]] += 1
            if max_val < dic[self.y[index]]:
                max_val = dic[self.y[index]]
                prediction = self.y[index]
        return prediction
            

        



knn = KNNClassifier(train_credit_data, 9)


predictions = []
for _, row in test_credit_data.iterrows():
    predictions.append(knn.predict(row))


knn_accuracy, knn_vals, knn_auc = confusion_matrix(predictions, test_credit_data["y"].to_numpy(), [1])


#  Decision Tree


credit_dt = DecisionTree(train_credit_data, max_depth=8, minimum_samples=5)


predictions = []
for _, row in test_credit_data.iterrows():
    predictions.append(credit_dt.predict(row))


credit_dt_accuracy, credit_dt_vals, credit_dt_auc = confusion_matrix(predictions, test_credit_data['y'].to_numpy(), [1])


compare_bar([lr_accuracy, credit_dt_accuracy, knn_accuracy], ["Logistic Regression", "Decision Tree", "KNN"], "Accuracy", "Techniques")
compare_bar([lr_vals[0][0], credit_dt_vals[0][0], knn_vals[0][0]], ["Logistic Regression", "Decision Tree", "KNN"], "Recall", "Techniques")
compare_bar([lr_vals[0][2], credit_dt_vals[0][2], knn_vals[0][2]], ["Logistic Regression", "Decision Tree", "KNN"], "F1 score", "Techniques")
compare_bar([lr_auc, credit_dt_auc, knn_auc], ["Logistic Regression", "Decision Tree", "KNN"], "ROC AUC", "Techniques")


#  Observations
# We see that all three classifiers have the same accuracy, however the decision tree has the best recall and f1 score, making it a better classifier. The Logistic Regression classifier has the worst recall and f1 score.


#  Training Random Forests using Bagging and Boosting ensemble methods


credit_rf_bagging = RandomForest(train_credit_data, [{"splitting_criteria":"gini", "max_depth":8, "minimum_samples":5} for _ in range(4)], p_row=0.8, handle_imbalance=True)
credit_rf_boosting = RandomForest(train_credit_data, [{"splitting_criteria":"gini"} for _ in range(10)], p_column=1, ensemble_method="boosting", handle_imbalance=True)


predictions_bag, predictions_boost = [], []
for _, row in test_credit_data.iterrows():
    predictions_bag.append(credit_rf_bagging.predict(row))
    predictions_boost.append(credit_rf_boosting.predict(row))


credit_rf_bag_acc, credit_rf_bag_vals, credit_rf_bag_auc = confusion_matrix(predictions_bag, test_credit_data['y'].to_numpy(), [1])


credit_rf_boost_acc, credit_rf_boost_vals, credit_rf_boost_auc = confusion_matrix(predictions_boost, test_credit_data['y'].to_numpy(), [1])


compare_bar([credit_rf_bag_acc, credit_rf_boost_acc], ["Bagging", "Boosting"], "Accuracy", "Techniques")
compare_bar([credit_rf_bag_vals[0][0], credit_rf_boost_vals[0][0]], ["Bagging", "Boosting"], "Recall", "Techniques")
compare_bar([credit_rf_bag_vals[0][1], credit_rf_boost_vals[0][1]], ["Bagging", "Boosting"], "F1 score", "Techniques")
compare_bar([credit_rf_bag_auc, credit_rf_boost_auc], ["Bagging", "Boosting"], "ROC AUC", "Techniques")


#  Observations
# We see that both ensemble methods have the same metrics with the boosting random forest having a slightly higher F1 score


compare_bar([credit_rf_bag_acc, credit_rf_boost_acc, credit_dt_accuracy, lr_accuracy, knn_accuracy], ["Bagging", "Boosting", "Decision Tree", "Logistic Regression", "K-NN"], "Accuracy", "Techniques")
compare_bar([credit_rf_bag_vals[0][0], credit_rf_boost_vals[0][0], credit_dt_vals[0][0], lr_vals[0][0], knn_vals[0][0]], ["Bagging", "Boosting", "Decision Tree", "Logistic Regression", "K-NN"], "Recall", "Techniques")
compare_bar([credit_rf_bag_vals[0][2], credit_rf_boost_vals[0][2], credit_dt_vals[0][2], lr_vals[0][2], knn_vals[0][2]], ["Bagging", "Boosting", "Decision Tree", "Logistic Regression", "K-NN"], "F1 score", "Techniques")
compare_bar([credit_rf_bag_auc, credit_rf_boost_auc, credit_dt_auc, lr_auc, knn_auc], ["Bagging", "Boosting", "Decision Tree", "Logistic Regression", "K-NN"], "ROC AUC", "Techniques")


#  Observations
# We see that that the ensemble models have better F1 scores and Recall with a slightly worse accuracy, making them the better classifiers.
# 
#  Strengths of Ensemble Approaches
# - Improved accuracy and performance
# - Reduce risk of overfitting and regularizes well
# - Provide more diversity
# - Highly versatile
# - Usually can be processed parallelly
#  Weaknesses of Ensemble Approaches
# - Computationally Expensive
# - Difficult to interpret
# - Sensitive to the quality of the base models and data


for depth in [2, 10]:
    for nt in [2, 4]:
        for sc in ['gini', 'entropy']:
            tt = RandomForest(train_credit_data, [{"max_depth":depth, "splitting_criteria":sc, "minimum_samples":3} for _ in range(nt)], p_row=0.6, handle_imbalance=True)
            predictions = []
            for _, row in test_credit_data.iterrows():
                predictions.append(tt.predict(row))

            print("Depth: %s, Number of trees: %s, Splitting criteria: %s"%(depth, nt, sc))
            confusion_matrix(predictions, test_credit_data['y'].to_numpy(), [1])


#  Observations
# We see that keeping the random forest simple is the best strategy, as the random forest with 2 trees and a maximum depth of 2 had the best F1 score and recall
