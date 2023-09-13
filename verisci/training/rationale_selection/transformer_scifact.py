import argparse
import torch
import jsonlines
import os


from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--claim-train', type=str, required=True)
parser.add_argument('--claim-dev', type=str, required=True)
parser.add_argument('--dest', type=str, required=True, help='Folder to save the weights')
parser.add_argument('--model', type=str, default='roberta-large')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch-size-gpu', type=int, default=8, help='The batch size to send through GPU')
parser.add_argument('--batch-size-accumulated', type=int, default=256, help='The batch size for each gradient update')
parser.add_argument('--lr-base', type=float, default=1e-5)
parser.add_argument('--lr-linear', type=float, default=1e-3)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("==============rationale_selection_train==================")
print(f'Using device "{device}"')


class SciFactRationaleSelectionDataset(Dataset):
    def __init__(self, corpus: str, claims: str):
        """
        - Phương thức kởi tạo 
        - Nhận hai tham số là corpus và claim, cả hai đều được dẫn tới các têp jsonl
        - Đọc nội dung của tệp corpus và lưu trữ chúng trong một dict, trong đó khóa là doc_id
        và giá trị là tài liệu tương ứng. Nó giả định rằng tệp corpus chứa một tệp JSONL với mỗi
        dòng đại diệ cho một tài liệu định dạng json, và mỗi tài liệu có một doc_id duy nhất
        - Lặp qua mỗi dòng trong tệp claims, cũng được giả đinh là một tệp JSONL. Đối với mỗi claim, nó truy 
        xuất bằng chứng liên quan từ trừng evidence, đó là một dict trong đó khóa là doc_id và giá trị là một 
        danh sách các câu chứng cứ.
        - Đối với mỗi câu chứng cứ, nó truy xuất tài liệu tương ứng từ dict corpus bằng cách
        sử dụng doc_id. Sau đó, nó tạo ra một tập hợp các chỉ số câu chứng cứ (evidence_sentence_idx) bằng cách trích
        xuất trường sentences từ mỗi múc chứng cứ. 
        """
        self.samples = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        for claim in jsonlines.open(claims):
            for doc_id, evidence in claim['evidence'].items():
                doc = corpus[int(doc_id)]
                evidence_sentence_idx = {s for es in evidence for s in es['sentences']}
                for i, sentence in enumerate(doc['abstract']):
                    self.samples.append({
                        'claim': claim['claim'],
                        'sentence': sentence,
                        'evidence': i in evidence_sentence_idx
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


trainset = SciFactRationaleSelectionDataset(args.corpus, args.claim_train)
devset = SciFactRationaleSelectionDataset(args.corpus, args.claim_dev)

tokenizer = AutoTokenizer.from_pretrained(args.model) # Tạo một toekenizer, để tiền xử lý văn bản trong quá trình huấn luyện và dự đoán
model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
# Tạo một mô hìn dự đoán chuỗi phân loại tuần tự bằng cách sử dụng tham số arg.model. Mô hình được tải từ các trọng số đã được huấn luyện trước (bert-large)

optimizer = torch.optim.Adam([
    {'params': model.roberta.parameters(), 'lr': args.lr_base},  # if using non-roberta model, change the base param path.
    {'params': model.classifier.parameters(), 'lr': args.lr_linear}
])
# Tạo trình tối ưu hóa Adam với hai nhóm tham số

scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 20)
# Lịch trình này được sử dụng để điều chỉnh tỉ lệ học của trình tối ưu hóa theo một lịch trình cosine với giai đoạn khởi động 20 vòng lặp


def encode(claims: List[str], sentences: List[str]):
    """
    - Hàm encode nhận vào hai danh sách là claims và sentences
    - Trong hàm, đầu tiên, tokenizer dược sử dụng để mã hóa cặp câu (sentence, claim). Mã hóa được thưc hiện 
    bằng các chuyển đổi các cặp câu thàn các mã đầu vào cho mô hình thông qua phương thức bathch_encode_plus. Giới hạn 512 
    - Sau đó, encoded_dict được chuyển đổi để chuyển đổi các tensor sang device được xác định trước đó 
    Việc này đảm bảo rằng các tensor sẽ đươc đặt trên cùng thiết bị mà mô hinhd đang sử 
    - Cuối cùng, encodded_dict được trả về từ hàm là một dict chứa các tensor đã được mã hóa
    """
    encoded_dict = tokenizer.batch_encode_plus(
        tuple(zip(sentences, claims)),
        pad_to_max_length=True,
        return_tensors='pt')
    if encoded_dict['input_ids'].size(1) > 512:
        # Too long for the model. Truncate it
        encoded_dict = tokenizer.batch_encode_plus(
            tuple(zip(sentences, claims)),
            max_length=512,
            truncation_strategy='only_first',
            pad_to_max_length=True,
            return_tensors='pt')
    encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
    return encoded_dict


def evaluate(model, dataset):
    model.eval()
    targets = []
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size_gpu):
            encoded_dict = encode(batch['claim'], batch['sentence'])
            logits = model(**encoded_dict)[0]
            targets.extend(batch['evidence'].float().tolist())
            outputs.extend(logits.argmax(dim=1).tolist())
    return f1_score(targets, outputs, zero_division=0),\
           precision_score(targets, outputs, zero_division=0),\
           recall_score(targets, outputs, zero_division=0)


for e in range(args.epochs):
    model.train()
    t = tqdm(DataLoader(trainset, batch_size=args.batch_size_gpu, shuffle=True))
    for i, batch in enumerate(t):
        encoded_dict = encode(batch['claim'], batch['sentence'])
        a = model(**encoded_dict, labels=batch['evidence'].long().to(device))
        loss = a.loss
        logits = a.logits
        loss.backward()
        if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
            optimizer.step()
            optimizer.zero_grad()
            t.set_description(f'Epoch {e}, iter {i}, loss: {round(loss.item(), 4)}')
    scheduler.step()
    train_score = evaluate(model, trainset)
    print(f'Epoch {e}, train f1: %.4f, precision: %.4f, recall: %.4f' % train_score)
    dev_score = evaluate(model, devset)
    print(f'Epoch {e}, dev f1: %.4f, precision: %.4f, recall: %.4f' % dev_score)
    # Save
    save_path = os.path.join(args.dest, f'epoch-{e}-f1-{int(dev_score[0] * 1e4)}')
    os.makedirs(save_path)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
