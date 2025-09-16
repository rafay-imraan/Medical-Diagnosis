# Medical-Diagnosis
An agent that asks the user a series of questions and predicts what disease they might have based on everything the model has learnt from a dataset downloaded from [Kaggle](https://www.kaggle.com/). This model was trained using the Random Forest algorithm provided by the [scikit-learn](https://scikit-learn.org/stable/index.html) library. Other scikit-learn functions used are [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) and [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html). [Pandas](https://pandas.pydata.org/) and [numpy](https://numpy.org/) was also used in the project. The agent asks the following questions:

- Do you have a fever?
- Do you have a cough?
- Are you feeling fatigue?
- Do you have difficulty breathing?
- Gender?
- Age?
- Blood Pressure?
- Cholestrol?

Then, it predicts what is the most likely disease out of a large number of diseases that it has learnt.

**MAKE SURE THE AFOREMENTIONED LIBRARIES ARE INSTALLED BEFORE EXECUTING THE CODE**
