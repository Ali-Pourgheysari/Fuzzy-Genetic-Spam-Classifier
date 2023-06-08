from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import random
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

    def test(self, rules, X_test): # 3, 4
        y_hat = []
        rules_with_class_0 = [rule for rule in rules if rule.class_label == 0]
        rules_with_class_1 = [rule for rule in rules if rule.class_label == 1]
        
        for row in X_test:
            g_0 = 0
            g_1 = 0
            for rule in rules_with_class_0:
                g_0 += (self.calculate_matching(rule.if_term, row))
            for rule in rules_with_class_1:
                g_1 += (self.calculate_matching(rule.if_term, row))
            y_hat.append(0 if g_0 > g_1 else 1)
        
        return y_hat

    def calculate_matching(self, if_term, row): # 2
        matching = 1

        for term in if_term: # [[4, 'fairly high', 'sigmoid', 93.50738309481642, 21], [1, 'medium', 'trapezius', 72.75815149671719, 59], [0, 'high', 'trapezius', 95.13684144852566, 6], [3, 'fairly low', 'sigmoid', 62.365788251728, -45]]
            if term[2] == 'sigmoid':
                matching *= self.sigmoid(row[term[0]], term[4], term[3])
            elif term[2] == 'gaussian':
                matching *= self.gaussian(row[term[0]], term[4], term[3])
            elif term[2] == 'triangular':
                matching *= self.triangular(row[term[0]], term[4], term[3])
            elif term[2] == 'trapezius':
                matching *= self.trapezius(row[term[0]], term[4], term[3])
    
        return matching
        

    def sigmoid(self, x, s, m):
        return 1 / (1 + np.exp(-(x-m)/s))
        

    def gaussian(self, x, s, m):
        return np.exp((-1/2)*((x-m)/s)**2)


    def trapezius(self, x, s, m):
        return max(min((x-m+s)/s, 1), 0)


    def triangular(self, x, s, m):
        return max(min((x-m+s)/s, (m-x+s)/s), 0)

    
class Rule:
    def __init__(self, maximum_value=None, minimum_value=None, is_offspring=False) -> None:
        if not is_offspring:
            self.if_term = self.generate_if_term(maximum_value, minimum_value)
            self.class_label = self.generate_class_label()
        self.fitness = None
    
    def generate_if_term(self, maximum_value, minimum_value):
        rule = []
        x_representive = [0, 1, 2, 3, 4]
        if_term_len = random.randint(1, 5)
        for j in range(if_term_len):
            term = []
            s = 0
            # representive of Xi
            term.append(random.choice(x_representive))
            x_representive.remove(term[0])
            term.append(random.choice(['low', 'fairly low', 'medium', 'fairly high', 'high'])) # term
            term.append(random.choice(['sigmoid', 'gaussian', 'triangular', 'trapezius'])) # membership function                 
            term.append(random.uniform(minimum_value, maximum_value)) # m
            while s == 0:
                s = random.uniform(1, abs(maximum_value - minimum_value)) if term[2] == 'triangular' else random.uniform(-1*abs(maximum_value - minimum_value), abs(maximum_value - minimum_value)) # s
            term.append(s)

            rule.append(term)

        return rule


    def generate_class_label(self):
        if random.uniform(0, 5) <= 4:
            return 0
        return 1


class genetic_algorithm:

    def __init__(self, X_train, y_train, population_len=50, generation=100, mutation_rate=0.1, crossover_rate=0.9, epsilon=1e-7) -> None:
        self.max_min_for_initial_generation(X_train)
        self.finess_score, self.parent_pool = self.algorithm(population_len, X_train, y_train, generation, mutation_rate, crossover_rate, epsilon)


    def max_min_for_initial_generation(self, X_train):
        flattened_arr = np.concatenate(X_train)
        negative_flattened_arr = np.array([x for x in flattened_arr if x < 0])
        positive_flattened_arr = np.array([x for x in flattened_arr if x >= 0])
        self.minimum_value = np.average(negative_flattened_arr)
        self.maximum_value = np.average(positive_flattened_arr)


    def algorithm(self, population_len, X_train, y_train, generation, mutation_rate, crossover_rate, epsilon):
        parent_pool = []
        for i in range(population_len):
            parent_pool.append(Rule(self.maximum_value, self.minimum_value))

        for rule in parent_pool:
            rule.fitness = self.fitness(rule, X_train, y_train)

        fitness_score = []

        last_total_fitness = 0
        for i in range(generation):
            fitness_score.append(np.average([rule.fitness for rule in parent_pool]))
            print(f'generation {i} average fitness: {fitness_score[i]}')

            if abs(fitness_score[i] - last_total_fitness) < epsilon:
                break
            last_total_fitness = fitness_score[i]
            
            parent_pool = self.selection(parent_pool, population_len)

            offspring = self.crossover(parent_pool, population_len, crossover_rate)

            offspring = self.mutation(offspring, population_len, mutation_rate)

            parent_pool += offspring

            for rule in parent_pool:
                rule.fitness = self.fitness(rule, X_train, y_train)

            parent_pool = self.selection(parent_pool, population_len)
        
        return fitness_score, parent_pool

    def selection(self, parent_pool, population_len):
        parents_with_label_0_len = 5*population_len//6
        parents_with_label_1_len = population_len - parents_with_label_0_len
        parent_pool = sorted(parent_pool, key=lambda x: x.fitness, reverse=True)
        parents_with_label_0 = [rule for rule in parent_pool if rule.class_label == 0]
        parents_with_label_1 = [rule for rule in parent_pool if rule.class_label == 1]
        parent_pool = parents_with_label_0[:parents_with_label_0_len] + parents_with_label_1[:parents_with_label_1_len]
        return parent_pool


    def crossover(self, parent_pool, population_len, crossover_rate):
        offspring = []
        parents_number = population_len//2 if population_len % 2 == 0 else population_len//2 - 1
        for i in range(parents_number - 1):
            if random.uniform(0, 1) < crossover_rate:
                first_child = Rule(is_offspring=True)
                second_child = Rule(is_offspring=True)
                first_child.if_term = parent_pool[i].if_term[:int(len(parent_pool[i].if_term)/2)] + parent_pool[i+1].if_term[int(len(parent_pool[i+1].if_term)/2):]
                second_child.if_term = parent_pool[i+1].if_term[:int(len(parent_pool[i+1].if_term)/2)] + parent_pool[i].if_term[int(len(parent_pool[i].if_term)/2):]
                if random.uniform(0, 1) < 0.5:
                    first_child.class_label, second_child.class_label = parent_pool[i].class_label, parent_pool[i+1].class_label
                else:
                    first_child.class_label, second_child.class_label = parent_pool[i+1].class_label, parent_pool[i].class_label
                offspring.append(first_child)
                offspring.append(second_child) 

        return offspring


    def mutation(self, offspring, population_len, mutation_rate):
        population_len = population_len//2 if population_len % 2 == 0 else population_len//2 - 1
        for i in range(population_len):
            if random.uniform(0, 1) < mutation_rate:
                if random.uniform(0, 1) < 0.5:
                    offspring[i].if_term = self.mutation_if_term(offspring[i].if_term)
                else:
                    offspring[i].class_label = 0 if offspring[i].class_label == 1 else 1

        return offspring


    def mutation_if_term(self, if_term):
        x_representive = [0, 1, 2, 3, 4]
        used_x_representive = []
        used_term_func_values = []
        for term in if_term:
            used_x_representive.append(term[0])
            used_term_func_values.append(term[1])

        if len(if_term) != 5:
            difference_x_representive = list(set(x_representive) - set(used_x_representive))
            if_term[random.randint(0, len(if_term) - 1)][0] = random.choice(difference_x_representive)

        # swap x representive
        random_number1 = random.randint(0, len(if_term) - 1)
        random_number2 = random.randint(0, len(if_term) - 1)
        if_term[random_number1][0], if_term[random_number2][0] = if_term[random_number2][0], if_term[random_number1][0]
        
        if_term[random.randint(0, len(if_term) - 1)][1] = random.choice([['low', 'fairly low', 'medium', 'fairly high', 'high']])
        random_number = random.randint(0, len(if_term) - 1)
        if_term[random_number][2] = random.choice(['sigmoid', 'gaussian', 'triangular', 'trapezius'])
        if_term[random.randint(0, len(if_term) - 1)][3] = random.uniform(self.minimum_value, self.maximum_value)
        s = 0
        while s == 0:
            s = random.uniform(1, abs(self.maximum_value - self.minimum_value)) if if_term[random_number][2] == 'triangular' else random.uniform(-1*abs(self.maximum_value - self.minimum_value), abs(self.maximum_value - self.minimum_value)) # s
        
        return if_term        


    def fitness(self, rule, X_train, y_train):
        if rule.fitness is None:
            fuzzy_functions = Fuzzy_functions()

            f_c = []
            f_neg = []
            X_with_same_class_label = [x for x, y in zip(X_train, y_train) if y == rule.class_label]
            X_with_different_class_label = [x for x, y in zip(X_train, y_train) if y != rule.class_label]

            for row in X_with_same_class_label:
                f_c.append(fuzzy_functions.calculate_matching(rule.if_term, row))
            s_f_c = sum(f_c)
            for row in X_with_different_class_label:
                f_neg.append(fuzzy_functions.calculate_matching(rule.if_term, row))
            s_f_neg = sum(f_neg)

            return (s_f_c - s_f_neg)/(s_f_c + s_f_neg) if s_f_c + s_f_neg != 0 else 0
        
        return rule.fitness
                
                
def main():
    data = data_preprocessing()
    X_train, X_test, y_train, y_test = train_test_split(data.records_dim_reduced, data.labels, test_size=0.33)
    ga = genetic_algorithm(X_train, y_train, population_len=50, generation=100, mutation_rate=0.1, crossover_rate=0.9, epsilon = 1e-7)
    fitness_score, rule_base = ga.finess_score, ga.parent_pool
    plt.plot(fitness_score)
    plt.show()

    ff = Fuzzy_functions()
    labels = ff.test(rule_base, X_test)
    counter = 0
    for cal_y, org_y in zip(labels, y_test):
        if cal_y == org_y:
            counter += 1
    print(f'accuracy is: {counter/len(y_test)}')
    

if __name__ == '__main__':
    main()