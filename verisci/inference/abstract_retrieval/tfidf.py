import argparse
import jsonlines
import numpy as np
from statistics import mean, median
from sklearn.feature_extraction.text import TfidfVectorizer

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--k', type=int, required=True)
parser.add_argument('--min-gram', type=int, required=True)
parser.add_argument('--max-gram', type=int, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

print("============tf-idf abstract retrival=====================")


# If we're doing the test data, don't evaluate.
run_evaluation = "test" not in args.dataset
"""
Kiểm tra xem liệu đang thực hiện đánh giá hay không bằng cách kiểm tra chuỗi test 
có trong tham số dataset hay không. Nếu không xuất hiện, run_evaliation sẽ 
có giá trị true, ngược lại là false.
"""

corpus = list(jsonlines.open(args.corpus))
dataset = list(jsonlines.open(args.dataset))
output = jsonlines.open(args.output, 'w')
# Đọc dữ liệu và đường dẫn lưu kết quả
k = args.k

vectorizer = TfidfVectorizer(stop_words='english',
                             ngram_range=(args.min_gram, args.max_gram))
# Tạo vectorizer, dùng để chuyển đổi văn bản thành vector TF_IDF

doc_vectors = vectorizer.fit_transform([doc['title'] + ' '.join(doc['abstract'])
                                        for doc in corpus])
# Vector title + abstract

doc_ranks = []

for data in dataset:
    claim = data['claim']
    claim_vector = vectorizer.transform([claim]).todense() # Chuyển đổi câu khẳng định thành vector TFIDF
    doc_scores = np.asarray(doc_vectors @ claim_vector.T).squeeze() # Tính điểm tương đồng giwax câu khẳng định và các tài liệu trong docvector bằng cách nhân ma trận và vector
    doc_indices_rank = doc_scores.argsort()[::-1].tolist() # Sắp xếp các chỉ số tài liệu theo thứ tự giảm dần 
    doc_id_rank = [corpus[idx]['doc_id'] for idx in doc_indices_rank]  # Tạo danh sách doc_id_rank bằng cách lấy các doc_id tương ứng với các chỉ số tài liệu tong doc_indeces_rank từ corpus

    if run_evaluation:
        for gold_doc_id in data['evidence'].keys():
            rank = doc_id_rank.index(int(gold_doc_id))
            doc_ranks.append(rank)

    output.write({
        'claim_id': data['id'],
        'doc_ids': doc_id_rank[:k]
    })
    # Ghi ra k abstract đầu tiên, mặc định là 3
if run_evaluation:
    print(f'Mid reciprocal rank: {median(doc_ranks)}')
    print(f'Avg reciprocal rank: {mean(doc_ranks)}')
    print(f'Min reciprocal rank: {min(doc_ranks)}')
    print(f'Max reciprocal rank: {max(doc_ranks)}')


print("============end tf-idf abstract retrival=====================")
