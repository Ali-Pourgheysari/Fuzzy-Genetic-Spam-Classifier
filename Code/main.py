from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

def process_data(sms_data_str):
    """
    convert `sms_data_str` into a pandas dataframe
    """
    data_arr = []

    data_records = sms_data_str.split('\n')[:-1]
    for data in data_records:
        label = None
        sample = None
        match data[:3]:
            case 'ham':
                label = 'legitimate'
                sample = data[4:] 
            case 'spa':
                label = 'spam'
                sample = data[5:] 
            case _:
                label = 'N/A'
            
        data_arr.append([label, sample])
        
    data_arr = np.array(data_arr)
    data_label = data_arr[:, 0]
    data_records = data_arr[:, 1]
    
    return data_records, data_label

def tfidf_vectorizer(records):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r'\b[A-Za-z]+\b', 
        norm=None
    )
    
    records_transformed = vectorizer.fit_transform(records)

    return records_transformed.toarray(), vectorizer.get_feature_names_out()

def feature_extraction(X, n_components=5):
    reduction_pca = PCA(
        n_components=n_components,
        whiten=False
    )
    data_reduced = reduction_pca.fit_transform(X)
    return data_reduced

def feature_selection(df_records, labels, n_components=5):
    feature_selection_model = SelectKBest(mutual_info_classif, k=n_components) 
    ## make a selection over the best features
    selected_record_features = feature_selection_model.fit_transform(df_records, labels)
    
    return selected_record_features, feature_selection_model.get_feature_names_out()

sms_data_str = None
with open('SMSSpamCollection') as file:
    sms_data_str = file.read()


records, labels = process_data(sms_data_str)
records_vectorized, feature_names = tfidf_vectorizer(records)

## one hot encoding labels
labels = np.array([0 if y == 'legitimate' else 1 for y in labels] )

## reducing dimension
records_dim_reduced = feature_extraction(records_vectorized)

# records_dim_reduced[:5]

records_vectorized = pd.DataFrame(records_vectorized, columns=feature_names)

records_selection, feature_name_selection = feature_selection(records_vectorized,labels=labels)

## for better visualization
pd.DataFrame(records_selection, columns=feature_name_selection).head()

## TODO: build a fuzzy rule-based model for (records, label) classification

# Define the Rule Base 
# Define linguistic variables and rules here
class Fuzzy_sets:
    # def max_min_of_features():
    #     maximum = records_dim_reduced[0][0]
    #     minimum = records_dim_reduced[0][0]

    #     for sublist in records_dim_reduced:
    #         for element in sublist:
    #             if element > maximum:
    #                 maximum = element
    #             if element < minimum:
    #                 minimum = element
        
    #     return minimum, maximum

    def sigmoid(x, s, m):
        return 1 / (1 + np.exp(-(x-m)/s))
        

    def gaussian(x, s, m):
        return np.exp((-1/2)*((x-m)/s)**2)


    def trapezius(x, s, m):
        return max(min((x-m)/s, 1), 0)


    def triangular(x, s, m):
        return max(min((x-m)/s, (m-x)/s), 0)


    # def generate_rule(self):
    #     # [5, -6.3, 9.8, 0.1, 99]
    #     # [[[low, s, m, sigmoid], [high, 9, 2, sigmoid], [medium, 5, 4, sigmoid], [low, 5, 4, sigmoid], [low, 5, 4], [high, 5, 4]]
    #     #  [[high, 5, 4], [medium, 5, 4], [low, 5, 4], [, 5, 4], [low, 5, 4], [high, 5, 4]]]]
    #     population = []

    #     minimum, maximum = records_dim_reduced.max, records_dim_reduced.min
    #     for i in range(50):
    #         rule = []
    #         for j in range(5):
    #             term = []
    #             s = 0
                
    #             term.append(np.random.choice(['sigmoid', 'gaussian', 'triangular', 'trapezius'])) # membership function
    #             term.append(np.random.randint(minimum, maximum)) # m      ########## we must calculate the min and max of each feature
    #             while s == 0:
    #                 term.append(np.random.randint(1, 100) if term[0] == 'triangular' else np.random.randint(-100, 100)) # s
    #             term.append(np.random.choice(['low', 'fairly low', 'medium', 'fairly high', 'high']))
            
    #             rule.append(term)

    #         population.append(rule)

    #     return population
    
class Rule:
    def __init__(self, population=50) -> None:
        self.if_term = self.generate_if_term(population)
        self.class_label = self.generate_class_label(population)
        self.fitness = None
        self.is_calculated = False
    
    def generate_if_term(population):
        minimum, maximum = records_dim_reduced.max, records_dim_reduced.min
        rule = []
        for j in range(5):
            term = []
            s = 0
            
            term.append(np.random.choice(['sigmoid', 'gaussian', 'triangular', 'trapezius'])) # membership function
            term.append(np.random.randint(minimum, maximum)) # m      ########## we must calculate the min and max of each feature
            while s == 0:
                term.append(np.random.randint(1, 100) if term[0] == 'triangular' else np.random.randint(-100, 100)) # s
            term.append(np.random.choice(['low', 'fairly low', 'medium', 'fairly high', 'high']))
        
            rule.append(term)
        return rule
    
    def generate_class_label(population):
        return np.random.randint(0, 1)



def genetic_algorithm():
    
    population = Rule(50)



def cross_over():
    pass
def mutation():
    pass
def fitness():
    pass