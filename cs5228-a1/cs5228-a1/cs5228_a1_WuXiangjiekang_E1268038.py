import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances



def clean(df_cars_dirty):
    """
    Handle all "dirty" records in the cars dataframe

    Inputs:
    - df_cars_dirty: Pandas dataframe of dataset containing "dirty" records

    Returns:
    - df_cars_cleaned: Pandas dataframe of dataset without "dirty" records
    """   
    
    # We first create a copy of the dataset and use this one to clean the data.
    df_cars_cleaned = df_cars_dirty.copy()

    #########################################################################################
    ### Your code starts here ###############################################################

  # 1. Replace negative values in 'no_of_owners' column with NaN
    df_cars_cleaned['no_of_owners'] = df_cars_cleaned['no_of_owners'].apply(lambda x: np.nan if x < 0 else x)
    # 2. Replace "XXXXX" with NaN in the 'curb_weight' column
    df_cars_cleaned['curb_weight'] = df_cars_cleaned['curb_weight'].replace('XXXXX', np.nan)
    # 3. Iterate 'manufactured' and handle incorrect values, any value larger than 2023 will be minus 100.
    def fix_year(year_str):
        year = int(year_str)
        if year > 2023:
            year -= 100
        return str(year)
    df_cars_cleaned['manufactured'] = df_cars_cleaned['manufactured'].apply(fix_year)
    
    ### Your code ends here #################################################################
    #########################################################################################
    
    return df_cars_cleaned



def handle_nan(df_cars_nan):
    """
    Handle all nan values in the cars dataframe

    Inputs:
    - df_cars_nan: Pandas dataframe of dataset containing nan values

    Returns:
    - df_cars_no_nan: Pandas dataframe of dataset without nan values
    """       

    # We first create a copy of the dataset and use this one to clean the data.
    df_cars_no_nan = df_cars_nan.copy()

    #########################################################################################
    ### Your code starts here ###############################################################
    # Fill 'make' and 'url' based on 'model'
    df_cars_no_nan['make'].fillna(df_cars_no_nan.groupby('model')['make'].transform('first'), inplace=True)
    df_cars_no_nan['url'].fillna(df_cars_no_nan.groupby('model')['url'].transform('first'), inplace=True)
    # After the above step there are still 6 NaN value for 'url', this is because are 'model' groups where all 'url' values are NaN, this code won't be able to fill those missing values because there is no non-null value to fill from within that group. Therefore fill these 6 'url' based on 'model' and fill NaN with a default value (e.g., 'default_url')
    df_cars_no_nan['url'].fillna('default_url', inplace=True)
    # Fill 'mileage' and 'price' with mean values 
    df_cars_no_nan['mileage'].fillna(df_cars_no_nan['mileage'].mean(), inplace=True)
    df_cars_no_nan['price'].fillna(df_cars_no_nan['price'].mean(), inplace=True)


    ### Your code ends here #################################################################
    #########################################################################################
    
    return df_cars_no_nan



def extract_facts(df_cars_facts):
    """
    Extract the facts as required from the cars dataset

    Inputs:
    - df_card_facts: Pandas dataframe of dataset containing the cars dataset

    Returns:
    - Nothing; you can simply us simple print statements that somehow show the result you
      put in the table; the format of the  outputs is not important; see example below.
    """       

    #########################################################################################
    ### Your code starts here ###############################################################

    # Toy example -- assume question: What is the total number of listings?
    # print('#listings: {}'.format(len(df_cars_facts)))
    # print()
    # 1) What are the lowest and the highest prices for which a car has been sold?
    lowest_price = df_cars_facts['price'].min()
    highest_price = df_cars_facts['price'].max()
    print(f'Lowest price: ${lowest_price}, Highest price: ${highest_price}')
    print()
    
    # 2) How many different car makes are in the whole dataset?
    unique_car_make = df_cars_facts['make'].nunique()
    print(f'Number of different car makes: {unique_car_make}')
    print()

    # 3) How many Toyota Corolla (manufactured before 2010) have been sold?
    number_Of_corolla_2010 = len(df_cars_facts[(df_cars_facts['make'] == 'toyota') & (df_cars_facts['model'] == 'corolla') & (df_cars_facts['manufactured'] < 2010)])
    print(f'Number of Toyota Corolla (manufactured before 2010) sold: {number_Of_corolla_2010}')
    print()
    
   # 4) What are the top-3 most sold car makes (give the car make and the number of sales)?
    top_3_car_makes = df_cars_facts['make'].value_counts().nlargest(3)
    print('Top-3 most sold car makes:')
    for make, sales in top_3_car_makes.items():
        print(f'{make}: {sales} sales')
    print()

    # 5) Which SUV car model has been sold the most (give the model and the number of sales)?
    suv_sales = df_cars_facts[df_cars_facts['type_of_vehicle'] == 'suv']
    most_sold_suv_model = suv_sales['model'].value_counts().idxmax()
    most_sold_suv_model_count = suv_sales['model'].value_counts().max()
    print(f'Most sold SUV car model: {most_sold_suv_model}, Number of sales: {most_sold_suv_model_count}')
    print()

    # 6) Which car make generated the highest overall sale when only considering low-powered cars (power < 60)?
    low_powered_cars = df_cars_facts[df_cars_facts['power'] < 60]
    highest_sale_make = low_powered_cars.groupby('make')['price'].sum().idxmax()
    highest_sale_total = low_powered_cars.groupby('make')['price'].sum().max()
    print(f'Car make with the highest overall sale for low-powered cars: {highest_sale_make}, Total sale: ${highest_sale_total}')
    print()
    
    # 7) Which midsize sedan has the highest power-to-engine_cap ratio
    midsize_sedans = df_cars_facts[df_cars_facts['type_of_vehicle'] == 'mid-sized sedan'].copy()
    
    # Convert 'power' and 'engine_cap' columns to string and then extract numbers
    midsize_sedans['power'] = midsize_sedans['power'].astype(str).str.extract('(\d+)').astype(float)
    midsize_sedans['engine_cap'] = midsize_sedans['engine_cap'].astype(str).str.extract('(\d+)').astype(float)
    
    # Calculate the power-to-engine_cap ratio
    midsize_sedans['power_to_engine_cap'] = midsize_sedans['power'] / midsize_sedans['engine_cap']
    
    # Find the entry with the highest ratio
    highest_ratio_entry = midsize_sedans[midsize_sedans['power_to_engine_cap'] == midsize_sedans['power_to_engine_cap'].max()]
    
    # Display the result
    print('Midsize sedan with the highest power-to-engine_cap ratio:')
    print(f'Make: {highest_ratio_entry["make"].values[0]}')
    print(f'Model: {highest_ratio_entry["model"].values[0]}')
    print(f'Year of Manufacturing: {highest_ratio_entry["manufactured"].values[0]}')
    print(f'Power-to-Engine_Cap Ratio (2 decimal precision): {highest_ratio_entry["power_to_engine_cap"].values[0]:.2f}')
    print()

    # 8) What is the correlation between resale price and mileage, and between resale price and engine_cap? (Pearson correlation)
    resale_mileage_correlation = df_cars_facts[['price', 'mileage']].corr(method='pearson').loc['price', 'mileage']
    resale_engine_cap_correlation = df_cars_facts[['price', 'engine_cap']].corr(method='pearson').loc['price', 'engine_cap']
    print(f'Correlation between resale price and mileage: {resale_mileage_correlation:.2f}')
    print(f'Correlation between resale price and engine_cap: {resale_engine_cap_correlation:.2f}')
    print()



    

    ### Your code ends here #################################################################
    #########################################################################################


    

def kmeans_init(X, k, c1=None, method='kmeans++'):
    
    """
    Calculate the initial centroids for performin K-Means

    Inputs:
    - X: A numpy array of shape (N, F) containing N data samples with F features
    - k: number of centroids/clusters
    - c1: First centroid as the index of the data point in X
    - method: string that specifies the methods to calculate centroids ('kmeans++' or 'maxdist')

    Returns:
    - centroid_indices: NumPy array containing k centroids, represented by the
      indices of the respective data points in X
    """   
    
    centroid_indices = []
    
    # If the index of the first centroid index c1 is not specified, pick it randomly
    if c1 is None:
        c1 = np.random.randint(0, X.shape[0])
        
    # Add selected centroid index to list
    centroid_indices.append(c1)      
    
    
        
    # Calculate and add c2, c3, ..., ck 
    while len(centroid_indices) < k:
        
        c = None
        
        #########################################################################################
        ### Your code starts here ###############################################################

        # Helper function to calculate distances between data points and current centroids
        def calculate_distances(X, centroids):
            distances = euclidean_distances(X, X[centroids])
            min_distances = np.min(distances, axis=1)
            return min_distances
        
        ## Remember to cover the 2 cases 'kmeans++' and 'maxdist'
        if method == 'kmeans++':
            # For 'kmeans++', pick the next centroid with probability proportional to distance squared
            distances = calculate_distances(X, centroid_indices)
            probabilities = distances ** 2 / np.sum(distances ** 2)
            c = np.random.choice(np.arange(X.shape[0]), p=probabilities)
        elif method == 'maxdist':
            # For 'maxdist', pick the next centroid as the farthest data point
            distances = calculate_distances(X, centroid_indices)
            c = np.argmax(distances)

        
        ### Your code ends here #################################################################
        #########################################################################################                
            
        centroid_indices.append(c)
    
    # Return list of k centroid indices as numpy array (e.g. [0, 1, 2] for K=3)
    return np.array(centroid_indices)
