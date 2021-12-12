import numpy as np
from numpy.core.fromnumeric import argmin

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
    N, D = x.shape
    centers = [p]

    while len(centers) < n_cluster:
        r = np.full(N, generator.rand())
        centroids = np.array([x[i] for i in centers])
        mapped_x = np.tile(x, (1, len(centers))).reshape(N, len(centers), D)
        mapped_centroids = np.tile(centroids.reshape(1, len(centers) * D), (N, 1)).reshape(N, len(centers), D)
        distances = np.einsum("ijk, ijk->ij", mapped_x - mapped_centroids, mapped_x - mapped_centroids)
        max_distances = np.array([max(distance) for distance in distances])
        argmax_distances = np.array([np.argmax(distances[i]) for i in range(len(distances))])
        probabilities = max_distances / np.array([np.sum(np.transpose(distances)[k]) for k in argmax_distances])
        cum_probs = np.cumsum(probabilities)
        selections = np.array([0 if n in centers else 1 for n in range(N)])

        if np.any(cum_probs * selections > r[0]):
            centers.append(np.where(cum_probs * selections > r[0])[0][0])
        # else:
        #     centers.append(np.argmax(distances * selections))

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
        centroids = np.array([x[i] for i in self.centers])
        objective = None
        assignments = None
        times = 0
        
        for t in range(self.max_iter):
            times += 1
            mapped_x = np.tile(x, (1, self.n_cluster)).reshape(N, self.n_cluster, D)
            mapped_centroids = np.tile(centroids.reshape(1, self.n_cluster * D), (N, 1)).reshape(N, self.n_cluster, D)
            distances = np.einsum("ijk, ijk->ij", mapped_x - mapped_centroids, mapped_x - mapped_centroids)
            
            # Update membership
            membership = np.array([[1 if c == np.argmin(distances[n]) else 0 for c in range(self.n_cluster)] for n in range(N)])

            # Update centers
            centroids = (np.transpose(membership) @ x) / (np.transpose(membership) @ np.ones(x.shape))

            assignments = membership @ np.arange(self.n_cluster)
            new_objective = np.sum(membership * distances)

            if objective != None and objective - new_objective <= self.e:
                break

            objective = new_objective
    
        return centroids, assignments, times

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
        centroids = np.array([x[i] for i in centroid_func(len(x), self.n_cluster, x, self.generator)])
        centroid_labels = np.ones(self.n_cluster)

        for t in range(self.max_iter):
            mapped_x = np.tile(x, (1, self.n_cluster)).reshape(N, self.n_cluster, D)
            mapped_centroids = np.tile(centroids.reshape(1, self.n_cluster * D), (N, 1)).reshape(N, self.n_cluster, D)
            distances = np.einsum("ijk, ijk->ij", mapped_x - mapped_centroids, mapped_x - mapped_centroids)
            
            # Update membership
            membership = np.array([[1 if c == np.argmin(distances[n]) else 0 for c in range(self.n_cluster)] for n in range(N)])

            # Update centers
            centroids = (np.transpose(membership) @ x) / (np.transpose(membership) @ np.ones(x.shape))

            new_objective = np.sum(membership * distances)

            if objective != None and objective - new_objective <= self.e:
                break

            objective = new_objective

        training_membership = np.array([[1 if c == y[n] else 0 for c in range(self.n_cluster)] for n in range(N)])
        centroid_labels = np.array([np.argmax(votes) for votes in np.transpose(membership) @ training_membership])
        
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
        mapped_x = np.tile(x, (1, self.n_cluster)).reshape(N, self.n_cluster, D)
        mapped_centroids = np.tile(self.centroids.reshape(1, self.n_cluster*D), (N, 1)).reshape(N, self.n_cluster, D)
        # Compute distances -> (N, self.n_cluster)
        distances = np.einsum("ijk, ijk->ij", mapped_x - mapped_centroids, mapped_x - mapped_centroids)
        
        return np.array([self.centroid_labels[np.argmin(distances[n])] for n in range(N)])



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
    W, L, D = image.shape
    N, K = W * L, code_vectors.shape[0]
    pixels = image.reshape(N, 3)

    mapped_pixels = np.tile(pixels, (1, K)).reshape(N, K, 3)
    mapped_code_vectors = np.tile(code_vectors.reshape(1, K * 3), (N, 1)).reshape(N, K, 3)
    distances = np.einsum("ijk, ijk->ij", mapped_pixels - mapped_code_vectors, mapped_pixels - mapped_code_vectors)

    new_pixels = np.array([code_vectors[np.argmin(distance)] for distance in distances])

    return new_pixels.reshape(image.shape)
