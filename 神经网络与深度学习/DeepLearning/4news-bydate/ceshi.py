from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# 相对路径（假设你的.py文件与数据文件夹在同一目录下）
train_path = './4news-bydate-train'
test_path = './4news-bydate-test'

# 加载数据
train_data = load_files(train_path, encoding='utf-8', decode_error='ignore')
test_data = load_files(test_path, encoding='utf-8', decode_error='ignore')

# 文本表示方法：Count + TF-IDF
vectorizer = CountVectorizer()
x_train_counts = vectorizer.fit_transform(train_data.data)
x_test_counts = vectorizer.transform(test_data.data)

tfidf = TfidfTransformer()
x_train_tfidf = tfidf.fit_transform(x_train_counts)
x_test_tfidf = tfidf.transform(x_test_counts)

# MultinomialNB
print("📌 MultinomialNB + TF-IDF:")
clf1 = MultinomialNB().fit(x_train_tfidf, train_data.target)
print(classification_report(test_data.target, clf1.predict(x_test_tfidf), target_names=test_data.target_names))

# BernoulliNB
print("📌 BernoulliNB + TF-IDF:")
clf2 = BernoulliNB().fit(x_train_tfidf, train_data.target)
print(classification_report(test_data.target, clf2.predict(x_test_tfidf), target_names=test_data.target_names))

# GaussianNB（注意需要稠密数组 + 标准化）
scaler = StandardScaler()
x_train_dense = scaler.fit_transform(x_train_tfidf.toarray())
x_test_dense = scaler.transform(x_test_tfidf.toarray())

print("📌 GaussianNB + TF-IDF:")
clf3 = GaussianNB().fit(x_train_dense, train_data.target)
print(classification_report(test_data.target, clf3.predict(x_test_dense), target_names=test_data.target_names))
