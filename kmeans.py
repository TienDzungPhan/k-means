import numpy as np

#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: a list of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    p = generator.randint(0, n) #this is the index of the first center
    #############################################################################
    # TODO: implement the rest of Kmeans++ initialization. To sample an example
	# according to some distribution, first generate a random number between 0 and
	# 1 using generator.rand(), then find the the smallest index n so that the 
	# cumulative probability from example 1 to example n is larger than r.
    #############################################################################
    centers = [x[p]]

    while len(centers) < n_cluster:
        r = generator.rand()
        D = []

        # Compute distances from each points to closest existing center
        for point in x:
            min_d = np.infty

            for center in centers:
                d = np.sum((point - center)**2)
                min_d = min(min_d, d)
            
            D.append(min_d)

        # Compute probability and select point
        for j in range(len(D)):
            if D[j] / np.sum(D) > r:
                centers.append(x[j])
                break

    # DO NOT CHANGE CODE BELOW THIS LINE
    return centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)



class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array, 
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0), 
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        ###################################################################
        # TODO: Update means and membership until convergence 
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################
        objective = None
        assignments = None
        times = 0

        for t in range(self.max_iter):
            times += 1
            membership = np.zeros((N, len(self.centers)))
            
            # Update membership
            for n in range(N):
                label = np.argmin(np.array([np.sum((x[n] - self.centers[k])**2) for k in range(len(self.centers))]))
                membership[n][label] = 1

            # Update centers
            self.centers = (np.transpose(membership) @ x) / (np.transpose(membership) @ np.ones(x.shape))

            assignments = membership @ np.arange(len(self.centers))
            new_objective = np.sum((membership @ self.centers - x)**2)

            if objective != None and objective - new_objective <= self.e:
                break

            objective = new_objective
    
        return self.centers, assignments, times

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented, 
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################
        objective = None
        membership = None
        centroids = centroid_func(len(x), self.n_cluster, x, self.generator)
        centroid_labels = np.ones(self.n_cluster)

        for t in range(self.max_iter):
            membership = np.zeros((N, self.n_cluster))
            
            # Update membership
            for n in range(N):
                label = np.argmin(np.array([np.sum((x[n] - centroids[k])**2) for k in range(self.n_cluster)]))
                membership[n][label] = 1

            # Update centers
            centroids = (np.transpose(membership) @ x) / (np.transpose(membership) @ np.ones(x.shape))

            new_objective = np.sum((membership @ centroids - x)**2)

            if objective != None and objective - new_objective <= self.e:
                break

            objective = new_objective

        centroid_labels = np.array([np.argmax(row) for row in np.transpose(membership) @ membership])
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored 
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################
        # Map x from (N, D) and centroids from (n_cluster, D) to (N, n_cluster, D)
        # mapped_x = np.tile(x, (1, self.n_cluster)).reshape(N, self.n_cluster, D)
        # mapped_centroids = np.tile(self.centroids.reshape(1, self.n_cluster*D), (N, 1)).reshape(N, self.n_cluster, D)
        # distances = (mapped_x - mapped_centroids)**2
        # print(np.argmin([(x[0] - self.centroids[k])**2 for k in range(self.n_cluster)]))
        return np.array([self.centroid_labels[np.argmin([np.sum((point - self.centroids[k])**2) for k in range(self.n_cluster)])] for point in x])



def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################
    
