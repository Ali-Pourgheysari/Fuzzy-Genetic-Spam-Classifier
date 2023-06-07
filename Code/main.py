from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

class data_preprocessing:

    def __init__(self) -> None:
        sms_data_str = None
        with open('SMSSpamCollection') as file:
            sms_data_str = file.read()


        self.records, self.labels = self.process_data(sms_data_str)
        self.records_vectorized, self.feature_names = self.tfidf_vectorizer(self.records)

        ## one hot encoding labels
        self.labels = np.array([0 if y == 'legitimate' else 1 for y in self.labels] )

        ## reducing dimension
        self.records_dim_reduced = self.feature_extraction(self.records_vectorized)

        # records_dim_reduced[:5]

        self.records_vectorized = pd.DataFrame(self.records_vectorized, columns=self.feature_names)

        # self.records_selection, self.feature_name_selection = self.feature_selection(self.records_vectorized,labels=self.labels)

        # ## for better visualization
        # pd.DataFrame(self.records_selection, columns=self.feature_name_selection).head()

    def process_data(self, sms_data_str):
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

    def tfidf_vectorizer(self, records):
        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r'\b[A-Za-z]+\b', 
            norm=None
        )
        
        records_transformed = vectorizer.fit_transform(records)

        return records_transformed.toarray(), vectorizer.get_feature_names_out()

    def feature_extraction(self, X, n_components=5):
        reduction_pca = PCA(
            n_components=n_components,
            whiten=False
        )
        data_reduced = reduction_pca.fit_transform(X)
        return data_reduced

    def feature_selection(self, df_records, labels, n_components=5):
        feature_selection_model = SelectKBest(mutual_info_classif, k=n_components) 
        ## make a selection over the best features
        selected_record_features = feature_selection_model.fit_transform(df_records, labels)
        
        return selected_record_features, feature_selection_model.get_feature_names_out()



class Fuzzy_functions:

    def fuzzification(self, rules, X_train): # 3, 4
        y_hat = []
        g_0 = []
        g_1 = []
        for rule in rules:
            rule_fitness = []

            for row in X_train:
                rule_fitness.append(self.calculate_matching(rule.if_term, row))

            avg = sum(rule_fitness)/len(rule_fitness)
            if rule.class_label == 0:
                g_0.append(avg)
            else:
                g_1.append(avg)
                
        y_hat.append(0 if sum(g_0) > sum(g_1) else 1)
        return y_hat

    ###################### it always calculate from the first [x1, x2] but we want [x3, x5] #############################
    def calculate_matching(self, if_term, row): # 2
        matching = 1

        for i, term in enumerate(if_term): # [[low, sigmoid, s, m], [high, triangular, s, m], ...]
            if term[1] == 'sigmoid':
                matching *= self.sigmoid(row[i], term[3], term[2])
            elif term[1] == 'gaussian':
                matching *= self.gaussian(row[i], term[3], term[2])
            elif term[1] == 'triangular':
                matching *= self.triangular(row[i], term[3], term[2])
            elif term[1] == 'trapezius':
                matching *= self.trapezius(row[i], term[3], term[2])
    
        return matching
        

    def sigmoid(self, x, s, m):
        return 1 / (1 + np.exp(-(x-m)/s))
        

    def gaussian(self, x, s, m):
        return np.exp((-1/2)*((x-m)/s)**2)


    def trapezius(self, x, s, m):
        return max(min((x-m)/s, 1), 0)


    def triangular(self, x, s, m):
        return max(min((x-m)/s, (m-x)/s), 0)

    
class Rule:
    def __init__(self, maximum_value, minimum_value) -> None:
        self.if_term = self.generate_if_term(maximum_value, minimum_value)
        self.class_label = self.generate_class_label()
        self.fitness = None
    
    def generate_if_term(self, maximum_value, minimum_value):
        rule = []
        membership_func_values = ['low', 'fairly low', 'medium', 'fairly high', 'high']
        for j in range(np.random.randint(1, 5)):
            term = []
            s = 0
            # <----(-93)------------------------------------(18)---->
            term.append(np.random.choice(membership_func_values)) # term
            membership_func_values.remove(term[0])
            term.append(np.random.choice(['sigmoid', 'gaussian', 'triangular', 'trapezius'])) # membership function                 
            term.append(np.random.uniform(minimum_value, maximum_value)) # m      ########## we must calculate the min and max of each feature
            while s == 0:
                s = np.random.uniform(1, abs(maximum_value - minimum_value)) if term[1] == 'triangular' else np.random.randint(-100, 100) # s
            term.append(s)
        
            rule.append(term)
        return rule
    
    def generate_class_label(self):
        return np.random.randint(0, 1)


class genetic_algorithm:

    def __init__(self, X_train, y_train, population=50):
        self.max_min_for_initial_generation(X_train)
        self.algorithm()

    def max_min_for_initial_generation(self, X_train):
        flattened_arr = np.concatenate(X_train)
        self.maximum_value = np.amax(flattened_arr)
        self.minimum_value = np.amin(flattened_arr)

    def algorithm(self, population):
        parent_pool = []
        for i in range(population):
            parent_pool.append(Rule(self.maximum_value, self.minimum_value))
        
        fuzzy = Fuzzy_functions()
        self.fitness(parent_pool)

    def cross_over():
        pass
    def mutation():
        pass

    def fitness(parent_pool):
        for item in parent_pool:
            if item.fitness is not None:
                pass


def main():
    data = data_preprocessing()
    X_train, X_test, y_train, y_test = train_test_split(data.records_dim_reduced, data.labels, test_size=0.33)
    genetic_algorithm(X_train, y_train, 50)

if __name__ == '__main__':
    main()