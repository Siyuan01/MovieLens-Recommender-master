"""
Hybrid filling Collaborative filtering.
"""
import operator
from operator import itemgetter
import math
from collections import defaultdict
from copy import deepcopy
import similarity
import utils
from utils import LogTime


class HybridFillingCF:
    """
    Hybrid filling Collaborative filtering.
    Top-N recommendation.
    """

    def __init__(self, p_sim_item=125, q_user_item=10, k_sim_user=20, n_rec_movie=6, dataset_name='ml-100k',
                 save_model=True):
        """
        Init with n_rec_movie.
        :return: None
        """
        print("HybridFillingCF start...\n")
        self.n_rec_movie = n_rec_movie
        self.trainset = None
        self.filledset = None
        self.pre_score = defaultdict(dict)
        self.save_model = save_model
        self.usernum, self.itemnum = (943, 1682) if dataset_name == 'ml-100k' else (6040, 3952)
        self.P = p_sim_item
        self.K = k_sim_user
        self.Q = q_user_item
        if p_sim_item > self.itemnum or q_user_item > self.itemnum or k_sim_user > self.usernum:
            raise ValueError("p/q/k is invalid parameter!")
        self.user_sim_mat = None
        self.movie_sim_mat = None
        self.user_mean = None
        self.model_name = 'K={}-P={}-Q={}'.format(self.K, self.P, self.Q)
        self.result = {}

    def fit(self, trainset):
        """
        Fit the trainset by filling the missing value.
        :param trainset: train dataset
        :return: None
        """
        model_manager = utils.ModelManager()
        self.get_usermean(trainset)
        try:
            self.movie_popular = model_manager.load_model('movie_popular')
            self.movie_count = model_manager.load_model('movie_count')
            self.trainset = model_manager.load_model('trainset')
            self.movie_sim_mat = model_manager.load_model(self.model_name + '-movie_sim_mat')
            self.user_sim_mat = model_manager.load_model(self.model_name + '-user_sim_mat')
            self.filledset = model_manager.load_model(self.model_name + '-filledset')
            print('User origin similarity model has saved before.\nLoad model success...\n')
        except OSError:
            print('No model saved before.\nTrain a new model...')
            self.trainset = trainset
            self.user_sim_mat, _, _ = similarity.calculate_user_cosine_similarity(trainset=trainset)
            self.fillMissingValue(trainset)
            # recompute the user_sim_mat for the recommend method.
            # self.user_sim_mat, _, _ = similarity.calculate_user_cosine_similarity(self.filledset)
            print('Train a new model success.')
            if self.save_model:
                model_manager.save_model(self.movie_popular, 'movie_popular')
                model_manager.save_model(self.movie_count, 'movie_count')
                model_manager.save_model(self.movie_sim_mat, self.model_name + '-movie_sim_mat')
                model_manager.save_model(self.filledset, self.model_name + '-filledset')
                model_manager.save_model(self.user_sim_mat, self.model_name + '-user_sim_mat')
            print('The new model has saved success.\n')
        self.get_usermean(self.filledset)

    def get_usermean(self, trainset=None):
        '''
        Get the mean value of each user
        :param self.trainset: The rating matrix.
        :return: the mean value of each user
        '''
        if self.user_mean:
            return
        if not trainset:
            trainset = self.trainset
        self.user_mean = {}
        for user, movies in trainset.items():
            self.user_mean[user] = sum(movies.values()) / (1.0 * len(movies))

    def fillMissingValue(self, trainset):
        """
        Fill the trainset with mean value.
        :param trainset: train dataset(dict)
        :return: filledset(dict)
        """
        print("Init the filledset")
        self.filledset = defaultdict(dict)
        for user, movies in trainset.items():
            for movie, rating in movies.items():
                self.filledset[user][movie] = trainset[user][movie]

        num = sum([len(movies) for user, movies in self.filledset.items()])
        print("now the number of the records is %d, the data sparsity level is %.2f\n" % (
            num, 1 - (num / (1.0 * self.itemnum * self.usernum))))
        print('Begin to fill the matrix based item.')

        # fill based item
        # movie_popular and movie_count should and only need to be computed once.
        self.movie_sim_mat, self.movie_popular, self.movie_count = \
            similarity.calculate_item_cosine_similarity(trainset=trainset)
        movie_similarity = {}
        for i in range(1, self.itemnum + 1):
            if str(i) in self.movie_sim_mat:
                movie_similarity[str(i)] = sorted(self.movie_sim_mat[str(i)].items(), key=operator.itemgetter(1),
                                                  reverse=True)
                if len(movie_similarity[str(i)]) > self.P:
                    movie_similarity[str(i)] = movie_similarity[str(i)][:self.P]
        for user, movies in trainset.items():
            # 从这开始改的
            sim_sum = defaultdict(float)
            for movie, rating in movies.items():
                for related_movie, similarity_factor in movie_similarity[movie]:
                    if related_movie in movies:
                        continue
                    if related_movie not in self.filledset[user]:
                        self.filledset[user][related_movie] = 0
                    self.filledset[user][related_movie] += (similarity_factor * rating)
                    sim_sum[related_movie] += similarity_factor
            for related_movie in self.filledset[user].keys():
                if related_movie not in movies:
                    self.filledset[user][related_movie] /= sim_sum[related_movie]
        print('fill the matrix based item success.')

        num = sum([len(movies) for user, movies in self.filledset.items()])
        print("now the number of the records is %d, the data sparsity level is %.2f\n" % (
            num, 1 - (num / (1.0 * self.itemnum * self.usernum))))

        filledset = deepcopy(self.filledset)

        # self.user_sim_mat, _, _ = similarity.calculate_user_cosine_similarity(self.filledset)
        print('Begin to fill the matrix based user.')
        # fill based user
        user_similarity = {}
        for user, movies in filledset.items():
            user_similarity[user] = sorted(self.user_sim_mat[user].items(), key=operator.itemgetter(1), reverse=True)
            # the len of user_similarity is between 0 -- self.usernum-1
            if len(user_similarity[user]) > self.K:
                user_similarity[user] = user_similarity[user][:self.K]

        for user, movies in filledset.items():
            # find the neighbors' rating of this user for the q items which have the max evaluation counts.
            item_not_graded = []
            # Find items that have not been graded
            for i in range(1, self.itemnum + 1):
                if str(i) not in movies:
                    item_not_graded.append(str(i))

            item_graded_counts = defaultdict(int)
            for neighbor_user, _ in user_similarity[user]:
                for item in item_not_graded:
                    if item in filledset[neighbor_user]:
                        item_graded_counts[item] += 1
            q_items = [i for i, _ in sorted(item_graded_counts.items(), key=operator.itemgetter(1), reverse=True)]
            if len(q_items) > self.Q:
                q_items = q_items[:self.Q]

            # only fill the rating of the q items for the user
            for item in q_items:
                up = 0
                down = 0
                for neighbor_user, sim in user_similarity[user]:
                    if item in filledset[neighbor_user]:
                        down += sim
                        up += (sim * filledset[neighbor_user][item])
                # do not fill if the first p neighboring items of the item are not rated
                if down != 0:
                    self.filledset[user][item] = up / (1.0 * down)
            '''
            下面是填充邻近用户共同评价最多的q个物品（应该加一个前提：u未评价过的），错误
            item_graded_counts = defaultdict(int)
            for neighbor_user, sim in user_similarity[user]:
                for neighbor_user_movie in filledset[neighbor_user]:
                    item_graded_counts[neighbor_user_movie] += 1
            '''
        print('fill the matrix based user success.')
        num = sum([len(movies) for user, movies in self.filledset.items()])
        print("now the number of the records is %d, the data sparsity level is %.2f\n" % (
            num, 1 - (num / (1.0 * self.itemnum * self.usernum))))

    def recommend(self, user):
        """
        Find K similar users and recommend N movies for the user.
        :param user: The user we recommend movies to.
        :return: the N best score movies
        """
        if not self.user_sim_mat or not self.n_rec_movie or \
                not self.trainset or not self.movie_popular or not self.movie_count:
            raise NotImplementedError('HfCF has not init or fit method has not called yet.')

        K = self.K
        N = self.n_rec_movie
        if user not in self.trainset:
            print('The user (%s) not in trainset.' % user)
            return
        # print('Recommend movies to user start...')

        sim_list = sorted(self.user_sim_mat[user].items(), key=itemgetter(1), reverse=True)[0:K]

        # Use alpha to change the influence of the user's neighbors.
        alpha = 1 / (1.0 * sum([similarity_factor for similar_user, similarity_factor in sim_list]))
        watched_movies = self.trainset[user]

        # Find items that have not been graded
        for i in range(1, self.itemnum + 1):
            if str(i) in watched_movies:
                continue
            self.pre_score[user][str(i)] = self.user_mean[user]
            for similar_user, similarity_factor in sim_list:
                # predict the user's "interest" for each movie
                if str(i) in self.filledset[similar_user]:
                    self.pre_score[user][str(i)] += (alpha * similarity_factor * (
                                self.filledset[similar_user][str(i)] - self.user_mean[similar_user]))

        res = [movie for movie, _ in sorted(self.pre_score[user].items(), key=itemgetter(1), reverse=True)[0:N]]
        return res

    def test(self, testset):
        """
        Test the recommendation system by recommending scores to all users in testset.
        :param testset: test dataset
        :return:
        """
        if not self.n_rec_movie or not self.trainset or not self.movie_popular or not self.movie_count:
            raise ValueError('HfCF has not init or fit method has not called yet.')
        # if not self.user_mean:
        #     self.get_usermean(self.filledset)
        self.testset = testset
        print('Test recommendation system start...')
        N = self.n_rec_movie
        #  varables for precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_movies = set()
        # varables for popularity
        popular_sum = 0

        # record the calculate time has spent.
        test_time = LogTime(print_step=1000)

        for i, user in enumerate(self.trainset):
            test_movies = self.testset.get(user, {})
            rec_movies = self.recommend(user)
            for movie in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
                popular_sum += math.log(1 + self.movie_popular[movie])
                # log steps and times.
            rec_count += N
            test_count += len(test_movies)
            # print time per 500 times.
            test_time.count_time()
        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        popularity = popular_sum / (1.0 * rec_count)

        sum_R = 0
        sum_M = 0
        testsize = 0
        for user, movies in self.testset.items():
            for movie, rating in movies.items():
                if movie in self.pre_score[user]:
                    pui = self.pre_score[user][movie]
                else:
                    pui = self.user_mean[user]
                sum_R += ((rating - pui) ** 2)
                sum_M += math.fabs(rating - pui)
                testsize += 1
            test_time.count_time()
        RMSE = math.sqrt(sum_R / float(testsize))
        MAE = sum_M / float(testsize)

        print('Test recommendation system success.')
        test_time.finish()

        print('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f\n' %
              (precision, recall, coverage, popularity))

        print('RMSE=%.4f\tMAE=%.4f\n' % (RMSE, MAE))
        self.result['RMSE'] = RMSE
        self.result['MAE'] = MAE
        self.result['precision'] = precision
        self.result['recall'] = recall
        self.result['coverage'] = coverage
        self.result['popularity'] = popularity

    def predict(self, testset):
        """
        Recommend movies to all users in testset.
        :param testset: test dataset
        :return: `dict` : recommend list for each user.
        """
        movies_recommend = defaultdict(list)
        print('Predict scores start...')
        # record the calculate time has spent.
        predict_time = LogTime(print_step=500)
        for i, user in enumerate(testset):
            rec_movies = self.recommend(user)  # type:list
            movies_recommend[user].append(rec_movies)
            # log steps and times.
            predict_time.count_time()
        print('Predict scores success.')
        predict_time.finish()
        return movies_recommend
