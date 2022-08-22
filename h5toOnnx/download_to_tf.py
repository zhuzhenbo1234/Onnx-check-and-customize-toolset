from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf
from sys import argv
import os


class MyModule(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
  def forward(self, input_ids):
    out = self.model(input_ids)

    # Parse response, which is nested like [output_0, [output_1, output_2, ...]] for gpt2
    response = {}
    response.update({"output_0":out[0]})
    i = 1
    for tensor in out[1]:
        key = "output_{}".format(i)
        response.update({key:tensor})
        i+=1


    return response

if len(argv) > 1:
    model_size = argv[1]
else:
    model_size = 'gpt2'

outpath = os.path.join(os.path.expanduser('~'), 
    "Documents", "Booste", "v3_serving", "models", model_size, "1")

if not os.path.exists:
    print("Making dir")
    os.makedirs(outpath, exist_ok=True)

print("Downloading {} and saving to {}...".format(model_size, outpath))

model = TFGPT2LMHeadModel.from_pretrained(model_size)
print(type(model))
print(model.summary())

print("Downloaded")
print("Testing...")
tokenizer = GPT2Tokenizer.from_pretrained(model_size)
tokens = tokenizer("Hi, this is", return_tensors='tf')
print(tokens)
out = model(tokens['input_ids'])
print(out[0])

print("Saving...")
module = MyModule(model)
tf.saved_model.save(module, outpath, signatures={"forward": module.forward})
print("Saved")

print("Testing loaded saved_model...")
imported = tf.saved_model.load(outpath)
loaded = imported.signatures["forward"]
out = loaded(tokens['input_ids'])

print(out["output_0"])
