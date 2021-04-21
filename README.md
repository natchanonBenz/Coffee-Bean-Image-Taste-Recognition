# Coffee Bean Image Taste Recognition


## Sperate a traning and testing set 80:20

```python
fea_train_G1, fea_test_G1, tar_train_G1, tar_test_G1 = train_test_split(feature_G1,G1_target, test_size=0.2, random_state=1)
fea_train_G2, fea_test_G2, tar_train_G2, tar_test_G2 = train_test_split(feature_G2,G2_target, test_size=0.2, random_state=1)
fea_train_G3, fea_test_G3, tar_train_G3, tar_test_G3 = train_test_split(feature_G3,G3_target, test_size=0.2, random_state=1)
fea_train_G4, fea_test_G4, tar_train_G4, tar_test_G4 = train_test_split(feature_G4,G4_target, test_size=0.2, random_state=1)

main_feature_train = np.concatenate((fea_train_G1, fea_train_G2,fea_train_G3,fea_train_G4), axis=0)
main_target_train=np.concatenate((tar_train_G1, tar_train_G2,tar_train_G3,tar_train_G4), axis=0)
main_feature_test = np.concatenate(( fea_test_G1,  fea_test_G2, fea_test_G3, fea_test_G4), axis=0)
main_target_test=np.concatenate((tar_test_G1, tar_test_G2,tar_test_G3,tar_test_G4), axis=0)
```

## Decision Tree Configuaration

```python
dTree = DecisionTreeClassifier(max_depth = 30, min_samples_leaf = 4, max_features = 7)
dTree.fit(main_feature_train,main_target_train)
```

## Evaluation


```python
ModelDT = dTree.predict(main_feature_test)
print("\nEvaluation of DTree Model : \n", classification_report(main_target_test, ModelDT))
```
