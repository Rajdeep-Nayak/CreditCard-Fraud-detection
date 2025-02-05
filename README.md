**Credit Card Fraud Detection using Logistic Regression**

**Name** - Rajdeep Nayak  
**Project** - Credit Card Fraud Detection  
**Skills** - Logistic Regression, Support Vector Machine, K Nearest Neighbours, F1 Score, ROC-AUC Curve, Data Visualization, Exploratory Data Analysis, Data Science application in Finance, Machine Learning  
**Tools** - Google Colab, Jupyter Notebooks, Python, Numpy, Pandas, Matplotlib, Seaborn, Sklearn  

---



## **The Dataset**

The data was taken from Kaggle: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The columns do not have physical significance directly visible since, as per Kaggle, the data was compressed using Principal Component Analysis (PCA) to protect the privacy of individuals while making a realistic scenario dataset available to the public.

---

## **Data Preprocessing and Visualization**

![image](https://user-images.githubusercontent.com/86561124/174430662-5a302491-d9ca-4705-b5c3-2d263563f564.png)

### **Correlations**

![image](https://user-images.githubusercontent.com/86561124/174430686-86a03acf-d2b0-4888-bd7b-ca0a13df524e.png)
![image](https://user-images.githubusercontent.com/86561124/174430691-a1e9a345-2924-4a7f-9160-daab07f58af4.png)

The columns do not seem to have correlations with each other but have a strong correlation with the Class and Time variables. This indicates that simple models may be sufficient, and neural networks might not be necessary.

### **Relation Between Target Variables and Columns**

(Various plots showing relationships between different columns and the target variable)

A plot between different columns and amounts, along with different colors for the target variable, shows that our output classes are separable by a linear boundary even when graphing variables alone. Hence, **Logistic Regression** is a suitable model to classify the data.

---

## **Class Imbalance in Dataset**

![image](https://user-images.githubusercontent.com/86561124/174430979-7d6dbbfa-8949-43bc-acdb-96daa1587309.png)

This shows that fraud cases are significantly fewer than non-fraud cases, which is expected in this dataset.

To address the imbalance, we can use undersampling or oversampling. Here, **SMOTE (Synthetic Minority Over-sampling Technique)** was used to counter the class imbalance.

![image](https://user-images.githubusercontent.com/86561124/174431044-03c757fa-758b-4a1d-a3d7-8746f07290e8.png)

---

## **Training the Model**

![image](https://user-images.githubusercontent.com/86561124/174431091-84847f62-0628-4f7f-83e4-b5d0c4295477.png)

A **Logistic Regression Model** was trained. The model was showing a convergence warning, so the number of iterations was increased to 150.

---

## **Results from Part 1**

![image](https://user-images.githubusercontent.com/86561124/174431703-3b4657d4-d52d-48d3-a43b-bb6bb982f3a4.png)

The **F1 Score** came out to be **0.99**, meaning the classifier performed well. It managed to catch **91 out of 101 frauds**, preventing fraud **90% of the time**.

The confusion matrix, precision, recall, and F1 score have been displayed for better understanding. The confusion matrix readings and the F1 score indicate the success of the model.

![image](https://user-images.githubusercontent.com/86561124/174431788-0c9feb90-de29-4477-a6ad-25c0f6481663.png)

---

## **Results from Part 2**

Additional raw code and findings have been uploaded to the repository.

- **Fraud Cases Are Time Independent**: Time can be dropped from the dataset.
- **Lower Dimension Visualization**: Visualizing the data in lower dimensions shows clear separability.
- **Choosing the Best Model**: After discussions with seniors, I decided to **undersample** the dataset instead of using synthetic data.
- **Focusing on Recall**: As a business decision, false positives (flagging a non-fraud transaction as fraud) are worse than false negatives. Customers would not want their credit card transactions to be incorrectly declined. Hence, recall was prioritized over F1 score.

Final Model: **Logistic Regression** was the best performer.

![image](https://github.com/ayush-agarwal-0502/Credit-Card-Fraud-Detection-ML/assets/86561124/e7a10968-4eac-4a9d-85f0-8a5cbbc91f67)

Other models also performed well, but Logistic Regression was chosen due to its interpretability and effectiveness.

---

## **Conclusion**

- **Logistic Regression is highly effective** in detecting fraud in credit card transactions.
- **Data preprocessing and handling class imbalance** are crucial for improving performance.
- **Business decisions should consider recall** to minimize false positives and improve customer experience.


