from transformers import ElectraTokenizer, ElectraForSequenceClassification
from datetime import datetime
import torch, itertools

class HLGDClassifier:
    def __init__(self, model_card="google/electra-base-discriminator", model_file="cls_elec_base_hlgd_0.74f1.bin", device="cuda"):
        self.device = device
        self.tokenizer = ElectraTokenizer.from_pretrained(model_card)
        self.model = ElectraForSequenceClassification.from_pretrained(model_card)
        self.model.to(self.device)

        print(self.model.load_state_dict(torch.load(model_file), strict=False))

    def preprocess(self, a1, a2):
        sep_tok = self.tokenizer.sep_token
        ha, hb = a1['headline'], a2['headline']
        day_diff = abs(a1['pubdate'] - a2['pubdate']).days
        return torch.LongTensor(self.tokenizer.encode(sep_tok+str(day_diff)+sep_tok+ha, hb))

    def predict(self, articles1, articles2):
        batch = [self.preprocess(a1, a2) for a1, a2 in zip(articles1, articles2)]
        input_ids = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0).cuda()

        model_output = self.model(input_ids=input_ids)
        probs = torch.nn.functional.softmax(model_output["logits"], dim=-1)
        return probs[:, 1].tolist()


if __name__ == "__main__":
    model_file = "/home/phillab/models/cls_elec_base_hlgd_0.74f1.bin"
    hlgd_model = HLGDClassifier(model_card="google/electra-base-discriminator", model_file=model_file)

    articles = [{"headline": "Blue Origin's first space tourist flight takes off on July 20th",
                "pubdate": datetime(2021, 5, 5)},
                {"headline": "You can bid for a seat on Blue Origin’s first human spaceflight on July 20",
                "pubdate": datetime(2021, 5, 5)},
                {"headline": "Peloton Recalls Treadmills After Injuries and a Child’s Death",
                "pubdate": datetime(2021, 5, 6)},
                {"headline": "Blue Origin Gets Potential Lifeline in NASA Lunar Lander Competition",
                "pubdate": datetime(2021, 5, 12)},
                ]
    pairs = list(itertools.combinations(articles, 2))
    predictions = hlgd_model.predict([p[0] for p in pairs], [p[1] for p in pairs])
    for pair, pred in zip(pairs, predictions):
        print("=====================")
        print(pair[0]["headline"])
        print("---")
        print(pair[1]["headline"])
        print("Classifier's prediction: %.3f" % pred)
