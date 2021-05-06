from transformers import GPT2LMHeadModel, AutoTokenizer
import torch

def pad(data, padval=0):
    return torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=padval)

class GeneratorSwapModel:
    def __init__(self, model_card="gpt2-medium", model_file="models/gpt2med_headline_gen_1.645.bin", device="cuda"):
        self.model = GPT2LMHeadModel.from_pretrained(model_card)
        self.tokenizer = AutoTokenizer.from_pretrained(model_card)
        self.tokenizer.pad_token = "!"
        self.start_id = self.tokenizer.bos_token_id

        self.device = device
        self.model.to(self.device)
        self.model.eval()
        print(self.model.load_state_dict(torch.load(model_file)))

    def score_pair(self, body_a, body_b, headline_a, headline_b):
        tokenizer_outs = self.tokenizer([body_a, body_b], return_tensors="pt", truncation=True, padding="longest")
        encs = tokenizer_outs["input_ids"][:, :300].to(self.device)

        swapped_headlines = [headline_b, headline_a]
        decs = [self.tokenizer.encode(dec, add_special_tokens=False) for dec in swapped_headlines]
        decs = [dec[:(30-1)] for dec in decs] # We cut short, but we want the end token at the end

        decs_inp = pad([torch.LongTensor([self.start_id]+dec) for dec in decs], padval=0).to(self.device)
        decs_out = pad([torch.LongTensor(dec+[self.tokenizer.eos_token_id]) for dec in decs], padval=-1).to(self.device)

        with torch.no_grad():
            model_out_enc = self.model(input_ids=encs, past_key_values=None)
            model_out_dec = self.model(input_ids=decs_inp, past_key_values=model_out_enc["past_key_values"])

        crit = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        loss = crit(model_out_dec["logits"].view(-1, self.tokenizer.vocab_size), decs_out.view(-1)).view(decs_out.shape)
        mask = (decs_inp != torch.LongTensor([0]).to(self.device)).float()
        non_pad_count = torch.sum(mask, dim=1)
        loss_per = torch.sum(loss, dim=1) / non_pad_count

        score = torch.sum(loss_per).item()
        prediction = 1 if score < 6.75 else 0
        return {"score": score, "prediction": prediction}


if __name__ == "__main__":
    import itertools

    swap_model = GeneratorSwapModel(model_file="/home/phillab/models/gpt2med_headline_gen_1.645.bin")

    articles = [{"headline": "Blue Origin's first space tourist flight takes off on July 20th",
                "body": """After years of development and more than a handful of delays along the way, Blue Origin plans to attempt the first official flight of its New Shepard spacecraft on July 20th.
                The company will offer one seat to the highest bidder of an online auction that starts today. Until May 19th, anyone can visit Blue Origin's website and place a private bid.
                After that date, the company will unseal the bids, allowing all involved to see how much money is at play.
                The entire process will then culminate on June 12th with a live auction to determine the winner of the seat.
                The company will donate the money it raises to its STEM-focused foundation, Club for the Future."""},
                {"headline": "You can bid for a seat on Blue Origin’s first human spaceflight on July 20",
                "body": """Jeff Bezos' Blue Origin is offering up one seat on the inaugural flight of its suborbital rocket New Shepard, set to take place July 20 —
                but instead of a fixed-price ticket sale, the seat will go to the highest bidder.
                It'll work like this: From May 5-19, bidders will be able to bid any amount on an auction website.
                From May 19, the bids will be made "unsealed," or made visible, and bidders must continually exceed the highest bid to remain in the running for the seat.
                Bidding will conclude June 12 with a live online auction.
                From Blue Origin's website, it looks like the overall flight will be relatively quick, with the craft reaching apogee, or its highest point, four minutes after takeoff.
                The capsule containing the astronauts (and the lucky bidder) will land 10 minutes after takeoff near its launch site in West Texas."""},
                {"headline": "Peloton Recalls Treadmills After Injuries and a Child’s Death",
                "body": """Peloton is recalling its Tread+ and Tread treadmills, the at-home fitness company said on Wednesday,
                less than a month after it fought the U.S. Consumer Product Safety Commission as it warned that dozens
                of injuries and one death of a child had been linked to the machines.
                The commission, which issued an “urgent warning” for the machines in April,
                urged people who own the treadmills to immediately stop using them.
                Peloton is offering a full refund for the $4,295 machine with a 32-inch touch screen
                that allows runnersto work out with the aid of instructors."""}
                ]
    for art_a, art_b in itertools.combinations(articles, 2):
        print("-------------------------------")
        print(art_a["headline"])
        print(art_b["headline"])

        model_output = swap_model.score_pair(art_a["body"], art_b["body"], art_a["headline"], art_b["headline"])
        print("Headlines describe the same underlying event" if model_output["prediction"] == 1 else "Headlines describe different events")
        print("Score: %.2f (need to be < 6.75 to be positive)" % (model_output["score"]))
