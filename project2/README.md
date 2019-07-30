### **[Capstone Project 2: Sentiment Analysis of Movie Reviews using a Deep Learning Convolutional Neural Network](https://github.com/ahrimhan/data-science-project/tree/master/project2)**

### Documention
1. [Final Report](https://github.com/ahrimhan/data-science-project/blob/master/project2/reports/capstone2_final_report.pdf)
2. [Presentation](https://github.com/ahrimhan/data-science-project/blob/master/project2/reports/capstone2_presentation.pdf)

### Jupyter Notebooks
1. [Vocabulary Dictionary](https://github.com/ahrimhan/data-science-project/blob/master/project2/vocab.ipynb)
2. [Cleaning Documents](https://github.com/ahrimhan/data-science-project/blob/master/project2/cleaning_document.ipynb)
3. [Building and Training Deep Learning Models](https://github.com/ahrimhan/data-science-project/blob/master/project2/deep_learning_CNN.ipynb)


**Problem Statment.** Sentiment analysis (or opinion mining) is the task of identifying and classifying
the sentiment expressed in a piece of text as being positive or negative. Given a bunch of text, sentiment
analysis classifies peoples opinions, appraisals, attitudes, and emotions toward products, issues,
and topics. The sentiment analysis has a wide range of applications in industry from forecasting market
movements based on sentiment expressed in news and blogs, identifying customer satisfaction and dissatisfaction
from their reviews and social media posts. It also forms the basis for other applications like recommender systems.

In recent years, there has been a remarkable performance improvement to Natural Language Processing
(NLP) problems by applying deep learning neural networks. 
Therefore, in this project, **I build
deep learning models using various parameters to classify the positive and negative movie reviews using the Convolutional Neural Nework.** I compare models and observe the parameters affecting the
performance in accuracy.

**Expected Beneficiaries.** Automated and accurate sentiment analysis techniques can be used to detect
fake reviews, news, or blogs and are becoming more and more important due to the huge impact on the
business markets. We provide the potential beneficiaries of this work.

* Businesses can find consumer opinions and emotions about their products and services.
* E-commerce companies, such as Amazon and Yelp, can identify fake reviews. There are cases when the results are not consistent from the actual looks and the sentiment analysis (automated
analysis). For example, if majority reviews are positive, but the sentiment analysis determines
that reviews should not be positive. Then the administrators should inspect the reviews manually
if those are fake or not. Fake reviews are not only damaging both competing companies and
customers, but they also lead to reduced trust in e-commerce companies and lower sales.
* Potential customers also can know the opinions and emotions of existing users. Customers also
can use the system based on the sentiment analysis and check if the reviews are reliable and
trustable before they make the decisions on buying products or services.

**Approach.** We prepare the movie review text data for classification with deep learning methods. We
obtain the large data set of the movie reviews. We clean the documents of text reviews by
removing punctuations, stopwords, stemming and removing non-frequent words to prevent a model from
overfitting. In this pre-processing of documents, we use the more sophisticated methods in the NLTK
python package.
To build a deep learning Convolutional Neural Network (CNN) model, we basically use the sequential
model of Keras.

1. First, the Embedding layer is located. There are two ways of setting the embedding layer: using
the pre-trained word embedding or training new embedding from scratch. For a pre-trained word
embedding, we use the GloVe (Global Vectors for Word Representation) embeddings.
2. Second, a series of convolution 1D Neural Network and pooling layers are added according to
typical CNN for text analysis. Then, after flattening layer, fully connected Dense layers are added.
Since this is a binary classification problem, we use the Sigmoid function as an activation function
for the final Dense layer. To prevent overfitting, we add Dropouts to deactivate neurons randomly,
which forces the network to learn a more balanced representation.
3. Finally, we make the different deep learning models by adjusting the parameters and will find the
best accurate model. We later will investigate the features affecting the accuracy.

**Results.** In the paper, the performance of machine learning models are in range of 67.42% to
88.89%. The model performed best in the cross validated Support Vector Machine (SVM) when concatenated
with bag of words representation. In this project, we generate the preliminary results, and the
best accuracy of our deep learning models based on CNN is 90.14%.