# -*- coding = utf-8 -*-
"""
Main function to build recommendation systems.

"""
import matplotlib.pyplot as plt
import utils
from ItemCF import ItemBasedCF
from LFM import LFM
from UserCF import UserBasedCF
from dataset import DataSet
from most_popular import MostPopular
from random_pred import RandomPredict
from utils import LogTime
from MeanValueCF import MeanValueCF
from ModeValueCF import ModeValueCF
from HfCF import HybridFillingCF

def run_model(dataset_name,model_name,random_seed=1, test_size=0.2, clean=False, n=10, k=20, p=125, q=10):
    print('*' * 70)
    print('\tThis is %s model trained on %s with test_size = %.2f' % (model_name, dataset_name, test_size))
    print('seed,k,p,q= ',random_seed,k,p,q)
    print('*' * 70 + '\n')
    model_manager = utils.ModelManager(dataset_name, test_size)
    try:
        trainset = model_manager.load_model('trainset')
        testset = model_manager.load_model('testset')
    except OSError:
        ratings = DataSet.load_dataset(name=dataset_name)
        trainset, testset = DataSet.train_test_split(ratings, random_seed=random_seed,test_size=test_size)
        model_manager.save_model(trainset, 'trainset')
        model_manager.save_model(testset, 'testset')
    '''Do you want to clean workspace and retrain model again?'''
    '''if you want to change test_size or retrain model, please set clean_workspace True'''
    model_manager.clean_workspace(clean)
    if model_name == 'UserCF':
        model = UserBasedCF(k_sim_user=k,n_rec_movie=n)
    elif model_name == 'ItemCF':
        model = ItemBasedCF(k_sim_movie=k)
    elif model_name == 'MeanValueCF':
        model = MeanValueCF(dataset_name=dataset_name)
    elif model_name == 'ModeValueCF':
        model = ModeValueCF(dataset_name=dataset_name)
    elif model_name == 'HfCF':
        model = HybridFillingCF(q_user_item=q, k_sim_user=k, p_sim_item=p, n_rec_movie=n, dataset_name=dataset_name)
    elif model_name == 'Random':
        model = RandomPredict()
    elif model_name == 'MostPopular':
        model = MostPopular()
    elif model_name == 'UserCF-IIF':
        model = UserBasedCF(use_iif_similarity=True)
    elif model_name == 'ItemCF-IUF':
        model = ItemBasedCF(use_iuf_similarity=True)
    elif model_name == 'LFM':
        # K, epochs, alpha, lamb, n_rec_movie
        model = LFM(10, 20, 0.1, 0.01, 10)
    else:
        raise ValueError('No model named ' + model_name)
    model.fit(trainset)

    # adjust the number of test user
    # user_list=[1, 100, 233, 666, 888]
    # user_list=list(range(1,943+1))  if dataset_name == 'ml-100k' else list(range(1,6040+1))
    # recommend_test(model, user_list)

    model.test(testset)
    # print(model.user_sim_mat)
    return model.result


def recommend_test(model, user_list):
    for user in user_list:
        recommend = model.recommend(str(user))
        print("recommend for userid = %s:" % user)
        print(recommend)
        print()


# def para_model(dataset_name, model_type1, model_type2=None, test_size=0.2, k=20, p=125, q=10,n=10):
#     main_time = LogTime(words="Main Function")
#     return run_model(model_type1, dataset_name, test_size, True,n=n,k=k,p=p,q=q)
#     main_time.finish()


if __name__ == '__main__':
    dataset_name = 'ml-100k'
    # dataset_name = 'ml-1m'
    model_type = 'UserCF'
    # model_type = 'UserCF-IIF'
    # model_type = 'ItemCF'
    # model_type = 'Random'
    # model_type = 'MostPopular'
    # model_type = 'ItemCF-IUF'
    # model_type = 'LFM'
    # model_type= 'MeanValueCF'
    # model_type = 'ModeValueCF'
    model_type = 'HfCF'
    test_size = 0.2

    # result = run_model(dataset_name, model_name=model_type, random_seed=1, test_size=test_size, clean=True, k=20,p=50, q=10, n=10)

    all_result=[]
    RMSE_list = []
    MAE_list=[]
    pre_list=[]
    # for i in range(1,6):
    #     result = run_model(dataset_name, model_name=model_type,random_seed=i, test_size=test_size, clean=True, k=20, p=125, q=10, n=10)
    #     all_result.append(result)
    #     RMSE_list.append(result['RMSE'])
    #     MAE_list.append(result['MAE'])
    #     pre_list.append(result['precision'])
    # print(sum(RMSE_list)/len(RMSE_list),RMSE_list)
    # print(sum(MAE_list)/len(MAE_list),MAE_list)
    # print(sum(pre_list)/len(pre_list),pre_list)
    #
    k_list=[5,10,20,40,60]
    leni=5
    for k in k_list:
        RMSE_sum=0
        MAE_sum=0
        pre_sum=0
        for i in range(1, 6):
            result=run_model(dataset_name, model_name=model_type, random_seed=i,test_size=test_size,clean=True, k=30, p=100, q=k,n=10)
            RMSE_sum += result['RMSE']
            MAE_sum+=result['MAE']
            pre_sum += result['precision']
        RMSE_list.append(RMSE_sum/leni)
        MAE_list.append(MAE_sum / leni)
        pre_list.append(pre_sum/leni)
    print(k_list,RMSE_list)
    print(k_list, MAE_list)
    print(k_list, pre_list)
    plt.plot(k_list,MAE_list)
    plt.show()