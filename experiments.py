import numpy as np
from generate_datasets import load_data as load
from joblib import delayed, Parallel
import json
from ref_methods import kdqtree, LDD, knn_post_hoc
from util import score_drift_localization

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def run_experiment(split,itr, dataset,add_dim):
    X,y,y_true = load(dataset,add_dim)
    res = []
    out = ""
    for model_name,method in [ \
            ("kdq-0.1-10-",lambda X,y,y_true:kdqtree(X,y,y_true,min_size=0.1,min_samps=10, theta = np.linspace(0,1,20))), 
            ("kdq-0.05-10-",lambda X,y,y_true:kdqtree(X,y,y_true,min_size=0.1,min_samps=10, theta = np.linspace(0,1,20))), 
            ("kdq-0.01-10-",lambda X,y,y_true:kdqtree(X,y,y_true,min_size=0.1,min_samps=10, theta = np.linspace(0,1,20))), 
            ("kdq-0.1-20-",lambda X,y,y_true:kdqtree(X,y,y_true,min_size=0.1,min_samps=20, theta = np.linspace(0,1,20))), 
            ("kdq-0.05-20-",lambda X,y,y_true:kdqtree(X,y,y_true,min_size=0.1,min_samps=20, theta = np.linspace(0,1,20))), 
            ("kdq-0.01-20-",lambda X,y,y_true:kdqtree(X,y,y_true,min_size=0.1,min_samps=20, theta = np.linspace(0,1,20))), 
            ("kdq-0.1-5-",lambda X,y,y_true:kdqtree(X,y,y_true,min_size=0.1,min_samps=5, theta = np.linspace(0,1,20))), 
            ("kdq-0.05-5-",lambda X,y,y_true:kdqtree(X,y,y_true,min_size=0.1,min_samps=5, theta = np.linspace(0,1,20))), 
            ("kdq-0.01-5-",lambda X,y,y_true:kdqtree(X,y,y_true,min_size=0.1,min_samps=5, theta = np.linspace(0,1,20))),
            ("knn-post-hoc-5-",lambda X,y,y_true:knn_post_hoc(X,y,y_true,k=5)), 
            ("knn-post-hoc-8-",lambda X,y,y_true:knn_post_hoc(X,y,y_true,k=8)), 
            ("knn-post-hoc-12-",lambda X,y,y_true:knn_post_hoc(X,y,y_true,k=12)),
            ("LDD-05-", lambda X,y,y_true:LDD(X,y,y_true,k=int(X.shape[0]*0.050), alpha=[0.005,0.01,0.025,0.05,0.075,0.1,0.15,0.2] )),
            ("LDD-07-", lambda X,y,y_true:LDD(X,y,y_true,k=int(X.shape[0]*0.075), alpha=[0.005,0.01,0.025,0.05,0.075,0.1,0.15,0.2] )),
            ("LDD-10-", lambda X,y,y_true:LDD(X,y,y_true,k=int(X.shape[0]*0.100), alpha=[0.005,0.01,0.025,0.05,0.075,0.1,0.15,0.2] )),
            ("LDD-15-", lambda X,y,y_true:LDD(X,y,y_true,k=int(X.shape[0]*0.150), alpha=[0.005,0.01,0.025,0.05,0.075,0.1,0.15,0.2] )),
            ("LDD-20-", lambda X,y,y_true:LDD(X,y,y_true,k=int(X.shape[0]*0.200), alpha=[0.005,0.01,0.025,0.05,0.075,0.1,0.15,0.2] )),
            ("our-knn-",lambda X,y,y_true:score_drift_localization(X,y,y_true,KNeighborsClassifier(n_neighbors=8),k=8,p=[0.005,0.01,0.025,0.05,0.075,0.1,0.15,0.2] )),
            ("our-dectree-",lambda X,y,y_true:score_drift_localization(X,y,y_true,DecisionTreeClassifier(criterion='gini',min_samples_leaf=10),k=10,p=[0.005,0.01,0.025,0.05,0.075,0.1,0.15,0.2] )),
            ("our-fortree-",lambda X,y,y_true:score_drift_localization(X,y,y_true,RandomForestClassifier(criterion='gini',min_samples_leaf=10),k=10,p=[0.005,0.01,0.025,0.05,0.075,0.1,0.15,0.2] )),
            ]:
        run_res = method(X,y,y_true)
        non_score = [(pval,scores) for pval,scores in run_res.items() if type(pval) is str]
        non_score_d = dict(non_score)
        for pval, scores in run_res.items():
            if type(pval) is not str:
                res.append( (model_name+"%.4f"%pval, dict(list(zip(["ACC","F1","MCC"],scores))+non_score)) )

                
                for field in [split,itr, "%s-%d"%(dataset,add_dim), model_name+"%.4f"%pval]+list(scores):
                    out += str(field)+";"
                for non_score_d_key in ["MODEL_ACC","MODEL_NLL","MODEL_TV"]:
                    out += (str(non_score_d[non_score_d_key]) if non_score_d_key in non_score_d.keys() else "")+";"
                out = out[:-1]
                out += "\n"
    print(out, end="")
    return dict(res)

if __name__ == "__main__":
    n_itr = 200
    
    split = 0
    setup_descs = []
    for itr in range(n_itr):
        for add_dim in [0,1,2,5,10]:
            for dataset in ["MNIST",("SEA",500,2),("SEA",150,2),("SEA",100,2),("RHP",500,2),("RHP",150,2),("RHP",100,2),("blob",3,50),("blob",9,50),("blob",18,50),("blob",18,100),("chess",5,5,1,50),("chess",5,5,3,50),("chess",5,5,5,50),("chess",5,5,3,100)]:
                setup_descs.append( (split,itr,add_dim,dataset) )
                split += 1
    
    print( *("split","itr","dataset","method","ACC","F1","MCC","MODEL_ACC","MODEL_NLL","MODEL_TV"), sep=";")
    results = Parallel(n_jobs=-2)(delayed(lambda split,itr,add_dim,dataset: {"split": split, "itr": itr, "additional_dimensions": add_dim, "dataset": dataset, "results": run_experiment(split,itr,dataset,add_dim)})(split,itr,add_dim,dataset) for split,itr,add_dim,dataset in setup_descs)
    
    with open(input("Save json to: "),"w") as f:
        json.dump(results, f)
    