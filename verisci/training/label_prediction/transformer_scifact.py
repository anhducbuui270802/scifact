import argparse
import torch
import jsonlines
import random
import os

from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
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
parser.add_argument('--lr-linear', type=float, default=1e-4)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=======================label-predict-train=====================")
print(f'Using device "{device}"')


class SciFactLabelPredictionDataset(Dataset):
    def __init__(self, corpus: str, claims: str):
        self.samples = [] # Tạo một danh sách rỗng sample cho đối tượng hiện tại 

        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        label_encodings = {'CONTRADICT': 0, 'NOT_ENOUGH_INFO': 1, 'SUPPORT': 2}

        for claim in jsonlines.open(claims):
            # Đọc từng đối tượng claim
            # Trong vòng lặp, đoạn mã if kiểm tra xem claim có chứng cứ hay không
            if claim['evidence']:
                # Nếu có, nó sẽ thực hiện các bước để thêm mẫu vào self.samples

                for doc_id, evidence_sets in claim['evidence'].items():
                    """
                    Vòng lặp duyệt qua mỗi căp khóa - giá trị trong claim[evidence]
                    """
                    doc = corpus[int(doc_id)] # Tìm tài liệu tướng ứng trong dict corpus

                    # Add individual evidence set as samples:
                    for evidence_set in evidence_sets:
                        rationale = [doc['abstract'][i].strip() for i in evidence_set['sentences']]
                        # Tạo ra một danh sách các đoạn trích dẫn từ doc['abstract'] dựa trên các câu có trong envidencee_set[sentences].

                        self.samples.append({
                            'claim': claim['claim'],
                            'rationale': ' '.join(rationale),
                            'label': label_encodings[evidence_set['label']]
                        })
                        """
                        Một mẫu tích cực được thêm vào self.semples bằng cách sử dụng thông tin từ chứng cứ 
                        hiện tại. Mẫu này bao gồm: claim, rationale, label.
                        """

                    # Add all evidence sets as positive samples
                    rationale_idx = {s for es in evidence_sets for s in es['sentences']}
                    rationale_sentences = [doc['abstract'][i].strip() for i in sorted(list(rationale_idx))]
                    self.samples.append({
                        'claim': claim['claim'],
                        'rationale': ' '.join(rationale_sentences),
                        'label': label_encodings[evidence_sets[0]['label']]  # directly use the first evidence set label
                        # because currently all evidence sets have
                        # the same label
                    })
                    """
                    Một mẫu tích cực khác được thêm vaò self.samples bằng cách sử dụng tất 
                    cả các tập chứng cứ của tài liệu hiện tại. Mẫu này cũng bao gồm các thành phần như trên 
                    rationale là sự kết hợp của tất các các đoạn trích dẫn từ các tập chứng cứ và lable 
                    được mã hóa từ tập chứng cứ đầu tiên
                    """

                    # Add negative samples
                    non_rationale_idx = (set(range(len(doc['abstract'])))) - rationale_idx
                    non_rationale_idx = list(non_rationale_idx) 
                    non_rationale_idx = random.sample(non_rationale_idx,
                                                      k=min(random.randint(1, 2), len(non_rationale_idx)))
                    non_rationale_sentences = [doc['abstract'][i].strip() for i in sorted(list(non_rationale_idx))]
                    self.samples.append({
                        'claim': claim['claim'],
                        'rationale': ' '.join(non_rationale_sentences),
                        'label': label_encodings['NOT_ENOUGH_INFO']
                    })
                    """
                    Một mẫu tiêu cực được được thêm vào samples bằng cách sử dụng các 
                    câu không thuộc tập chứng cứ. Các câu này lựa chọn ngẫu nhiên từ tài liệu 
                    và được lưu trữ trong biến non_rationale_sentences.
                    Gồm các phần: claim, rationale, label. Với rationale là sự kết hợp của các câu không 
                    thuộc tập chứng cứ và label là NEI
                    """
            else:
                # Add negative samples
                # Thêm các mẫu tiêu cực vào samples 
                for doc_id in claim['cited_doc_ids']:
                    doc = corpus[int(doc_id)]
                    non_rationale_idx = random.sample(range(len(doc['abstract'])), k=random.randint(1, 2))
                    non_rationale_sentences = [doc['abstract'][i].strip() for i in non_rationale_idx]
                    # Chọn ngẫu nhiên một số chỉ mục của các câu không thuộc tập chứng cứ từ tài liệu. Số lượng các câu này được chọn ngẫu nhiên từ 1 - 2
                    # Sau đóa, các câu không thuộc tập chứng cứ được lưu trong biến non_rationale_sentences bằng chỉ mục lúc trước
                    self.samples.append({
                        'claim': claim['claim'],
                        'rationale': ' '.join(non_rationale_sentences),
                        'label': label_encodings['NOT_ENOUGH_INFO']
                    })
                    """
                    Một mẫu tiêu cực được thêm vào samples bằng cách sử dụng các câu không thuộc tập chứng cứ.
                    Mẫu này bao gồm các trường: claim, rationale, lable.
                    """

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


trainset = SciFactLabelPredictionDataset(args.corpus, args.claim_train)
devset = SciFactLabelPredictionDataset(args.corpus, args.claim_dev)

tokenizer = AutoTokenizer.from_pretrained(args.model)
config = AutoConfig.from_pretrained(args.model, num_labels=3)
model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config).to(device)
optimizer = torch.optim.Adam([
    # If you are using non-roberta based models, change this to point to the right base
    {'params': model.roberta.parameters(), 'lr': args.lr_base},
    {'params': model.classifier.parameters(), 'lr': args.lr_linear}
])
scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 20)


def encode(claims: List[str], rationale: List[str]):
    encoded_dict = tokenizer.batch_encode_plus(
        tuple(zip(rationale, claims)),
        pad_to_max_length=True,
        return_tensors='pt')
    if encoded_dict['input_ids'].size(1) > 512:
        # Too long for the model. Truncate it
        encoded_dict = tokenizer.batch_encode_plus(
            tuple(zip(rationale, claims)),
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
            encoded_dict = encode(batch['claim'], batch['rationale'])
            logits = model(**encoded_dict)[0]
            targets.extend(batch['label'].float().tolist())
            outputs.extend(logits.argmax(dim=1).tolist())
    return {
        'macro_f1': f1_score(targets, outputs, zero_division=0, average='macro'),
        'f1': tuple(f1_score(targets, outputs, zero_division=0, average=None)),
        'precision': tuple(precision_score(targets, outputs, zero_division=0, average=None)),
        'recall': tuple(recall_score(targets, outputs, zero_division=0, average=None))
    }


for e in range(args.epochs):
    model.train()
    t = tqdm(DataLoader(trainset, batch_size=args.batch_size_gpu, shuffle=True))
    for i, batch in enumerate(t):
        encoded_dict = encode(batch['claim'], batch['rationale'])
        a = model(**encoded_dict, labels=batch['label'].long().to(device))
        loss = a.loss
        logits = a.logits
        loss.backward()
        if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
            optimizer.step()
            optimizer.zero_grad()
            t.set_description(f'Epoch {e}, iter {i}, loss: {round(loss.item(), 4)}')
    scheduler.step()
    # Eval
    train_score = evaluate(model, trainset)
    print(f'Epoch {e} train score:')
    print(train_score)
    dev_score = evaluate(model, devset)
    print(f'Epoch {e} dev score:')
    print(dev_score)
    # Save
    save_path = os.path.join(args.dest, f'epoch-{e}-f1-{int(dev_score["macro_f1"] * 1e4)}')
    os.makedirs(save_path)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
