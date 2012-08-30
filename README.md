AdaBoost
========

# 計算数理応用の課題

# PAC学習

ある学習問題に対して

* 「誤識別確率がε以下である」確率が 1-δ 以上であるである

を満たす仮説を出力できるアルゴリズムが存在するとき、この学習問題はPAC
学習可能 (Probably Approximately Correct learnable) であるという。

# AdaBoost

弱仮説（弱学習器）を集めていい仮説作ろう！
必ず誤差0.5より小さい弱仮説があるならPAC学習可能。
具体的にAdaBoostをするにはどうやって弱学習器を選ぶかとか決める必要がある。


# プログラム

## 問題 識別問題
* 学習の対象 識別器
* 入力データ 事例集合
* 学習の目的 事例を説明する仮説を生成する

## 手法 以下の4つ
1. naive Bayes
2. Disj Learner
3. k-DL Learner
4. AdaBoost






