# Localization of Concept Drift: Identifying the Drifting Datapoints 

Experimental code of conference paper. [Paper](https://ieeexplore.ieee.org/document/9892374)

## Abstract

The notion of concept drift refers to the phenomenon that the distribution which is underlying the observed data changes over time. As a consequence machine learning models may become inaccurate and need adjustment. While there do exist methods to detect concept drift, to find change points in data streams, or to adjust models in the presence of observed drift, the problem of localizing drift, i.e. identifying it in data space, is yet widely unsolved -- in particular from a formal perspective. This problem however is of importance, since it enables an inspection of the most prominent characteristics, e.g. features, where drift manifests itself and can therefore be used to make informed decisions, e.g. efficient updates of the training set of online learning algorithms, and perform precise adjustments of the learning model. In this paper we present a general theoretical framework that reduces drift localization to a supervised machine learning problem. We construct a new method for drift localization thereon and demonstrate the usefulness of our theory and the performance of our algorithm by comparing it to other methods from the literature. 

## Requirements

* Python 
* Numpy, SciPy, Matplotlib
* scikit-learn

## Cite

Cite our Paper
```
@inproceedings{hinder2022localization,
  title={Localization of concept drift: Identifying the drifting datapoints},
  author={Hinder, Fabian and Vaquet, Valerie and Brinkrolf, Johannes and Artelt, Andr{\'e} and Hammer, Barbara},
  booktitle={2022 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--9},
  year={2022},
  organization={IEEE}
}
```
