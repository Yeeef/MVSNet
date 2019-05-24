# code reading

## overview

- `PlaneParamEstimator` 封装了对平面的估计，包括 exact / approximate plane fitting, 以及 agree 方法用来模糊检验一个点是否在平面上（基于阈值 $\epsilon$)
- `Ransac` 封装了整个 ransac 的方法，`compute` 函数就在做这件事情
  - `compute` 函数有两种实现
    - 第一种比较粗暴，它会直接检查所有可能性，找一个 inlier 最多的配置，核心过程写成了一个递归
    - 第二种利用统计知识算出一个计算次数的期望，减少搜索次数

## bruteforce compute

## statistical compute

## USELESS COLLECTIONS

- curSubSetIndexes
- chosenSubSets

上面这俩是为了确认选到的点和之前的没有重复

## Q&A

- [ ] numVotesForBest >= 10 就不找了？

