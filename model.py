# A file that creates 4 cluster models for the zillow dataset for independent features
#in terms of correlation to logerror.
from sklearn.cluster import KMeans


def create_clusters(train_scaled, validate_scaled, test_scaled):
    '''
    Function creates four clusters from scaled train - Features, SQFT, Rooms
    Fits KMeans to train, predicts on train, validate, test to create clusters for each.
    Appends clusters to scaled data for modeling.
    '''

    # Cluster_Features
    # Selecting Features
    X_1 = train_scaled[['calculatedfinishedsquarefeet', 'yearbuilt', 'lotsizesquarefeet','is_Los_Angeles','is_Ventura','is_Orange']]
    X_2 = validate_scaled[['calculatedfinishedsquarefeet', 'yearbuilt', 'lotsizesquarefeet','is_Los_Angeles','is_Ventura','is_Orange']]
    X_3 = test_scaled[['calculatedfinishedsquarefeet', 'yearbuilt', 'lotsizesquarefeet','is_Los_Angeles','is_Ventura','is_Orange']]
    # Creating Object
    kmeans = KMeans(n_clusters=3)
    # Fitting to Train Only
    kmeans.fit(X_1)
    # Predicting to add column to train
    train_scaled['cluster_features'] = kmeans.predict(X_1)
    # Predicting to add column to validate
    validate_scaled['cluster_features'] = kmeans.predict(X_2)
    # Predicting to add column to test
    test_scaled['cluster_features'] = kmeans.predict(X_3)

    # Property Sqft and Years built::
    #naming features as X for clustering
    X_4 = train_scaled[['calculatedfinishedsquarefeet', 'yearbuilt']]
    X_5 = validate_scaled[['calculatedfinishedsquarefeet', 'yearbuilt']]
    X_6 = test_scaled[['calculatedfinishedsquarefeet', 'yearbuilt']]
    # Creating Object
    kmeans = KMeans(n_clusters=3)
    # Fitting to Train Only
    kmeans.fit(X_4)
    # Predicting to add column to train
    #storing this predicted cluster of data into original dataframe
    train_scaled['cluster_prop_age_size'] = kmeans.predict(X_4)
    validate_scaled['cluster_prop_age_size'] = kmeans.predict(X_5)
    test_scaled['cluster_prop_age_size'] = kmeans.predict(X_6)

    # Is_Los_Angeles Cluster
    # Selecting Features
    #naming features as X for clustering
    X_7= train_scaled[['calculatedfinishedsquarefeet', 'yearbuilt','is_Los_Angeles','bathroomcnt','taxvaluedollarcnt','latitude','longitude']]
    X_8= validate_scaled[['calculatedfinishedsquarefeet', 'yearbuilt','is_Los_Angeles','bathroomcnt','taxvaluedollarcnt','latitude','longitude']]
    X_9= test_scaled[['calculatedfinishedsquarefeet', 'yearbuilt','is_Los_Angeles','bathroomcnt','taxvaluedollarcnt','latitude','longitude']]
    # Creating Object
    kmeans = KMeans(n_clusters=3)
    # Fitting to Train Only
    kmeans.fit(X_7)
    # Predicting to add column to train
    train_scaled['cluster_la'] = kmeans.predict(X_7)
    validate_scaled['cluster_la'] = kmeans.predict(X_8)
    test_scaled['cluster_la'] = kmeans.predict(X_9)

    # Cluster_Value: Calculated Sqft & TaxValue
    # Selecting Features
    #naming features as X for clustering
    X_10= train_scaled[['calculatedfinishedsquarefeet', 'taxvaluedollarcnt','latitude','longitude']]
    X_11= validate_scaled[['calculatedfinishedsquarefeet', 'taxvaluedollarcnt','latitude','longitude']]
    X_12= test_scaled[['calculatedfinishedsquarefeet', 'taxvaluedollarcnt','latitude','longitude']]
    # Creating Object
    kmeans = KMeans(n_clusters=3)
    # Fitting to Train Only
    kmeans.fit(X_10)
    # Predicting to add column to train
    train_scaled['cluster_value'] = kmeans.predict(X_10)
    validate_scaled['cluster_value'] = kmeans.predict(X_11)
    test_scaled['cluster_value'] = kmeans.predict(X_12)

    return train_scaled, validate_scaled, test_scaled