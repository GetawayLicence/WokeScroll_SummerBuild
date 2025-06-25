from flask import Blueprint, request, jsonify
from transformers import BertTokenizerFast, AutoModel
import torch
import torch.nn as nn

fake_news_bp = Blueprint("fake_news", __name__)

# Model Architecture
class BERT_Architecture(nn.Module):
  def __init__(self, bert):
    super(BERT_Architecture, self).__init__()
    self.bert = bert
    self.dropout = nn.Dropout(0.1)                            # Dropout Layer
    self.relu = nn.ReLU()                                     # Activation Layer
    self.fc1 = nn.Linear(768, 512)                            # Layer 1
    self.fc2 = nn.Linear(512, 2)                              # Layer 2
    self.softmax = nn.LogSoftmax(dim = 1)                     # Softmax Layer

  def forward(self, sent_id, mask):
    cls_hs = self.bert(sent_id, attention_mask = mask)['pooler_output'] # Inputs

    x = self.fc1(cls_hs)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.softmax(x)
    return x

# Load model and tokenizer
model_path = "models/fakenews_model_best_weights.pt"
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
try:
    bert = AutoModel.from_pretrained("bert-base-uncased")
    model = BERT_Architecture(bert)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
except Exception as e:
    print(f"Model loading failed: {e}")
    raise
model.eval()

MAX_LENGTH = 20  # must match your training setup

@fake_news_bp.route("/predict", methods=["POST"])
def predict_fake_news():
    data = request.get_json()
    title = data.get("title", "")

    if not title:
        return jsonify({"error": "No title provided"}), 400

    # Tokenizing input given
    inputs = tokenizer.batch_encode_plus(
        [title],
        max_length=MAX_LENGTH,
        padding = True,
        truncation = True,
        return_tensors = "pt"
    )

    with torch.no_grad():
        output = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.exp(output)
        confidence, predicted_class = torch.max(probs, dim=1)

    label = "Fake" if predicted_class.item() == 1 else "Real"  # adjust if needed

    return jsonify({
        "title": title,
        "prediction": label,
        "confidence": round(confidence.item(), 4)
    })