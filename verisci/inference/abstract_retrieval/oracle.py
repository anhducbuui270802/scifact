import argparse
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True) # dataset
parser.add_argument('--include-nei', action='store_true') # Nếu trong dữ liệu có NEI thì thêm cái này  
parser.add_argument('--output', type=str, required=True) # path file đầu ra
args = parser.parse_args()

dataset = jsonlines.open(args.dataset)
output = jsonlines.open(args.output, 'w')

print("============oracle abstract retrival=====================")

for data in dataset:
    doc_ids = list(map(int, data['evidence'].keys())) # Chuyển đổi các khóa của dict envidence thành số nguyên
    if not doc_ids and args.include_nei:
        doc_ids = [data['cited_doc_ids'][0]]
        # Nếu doc_ids rỗng và nei được chỉ định , thêm cited_ids đầu tiên của dữ liệu vào doc_ids

    output.write({
        'claim_id': data['id'],
        'doc_ids': doc_ids
    })

    # Ghi một đối tượng json mới 


print("============end oracle abstract retrival=================")
