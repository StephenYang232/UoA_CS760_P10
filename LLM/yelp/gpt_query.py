import os
from time import sleep
import json
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

infile = 'yelp_gpt_test.jsonl'
with open(infile, encoding="UTF-8") as f:
  inputs = [json.loads(line) for line in f]

outfile = 'gpt_output_baseline.csv'
# outfile = 'gpt_output_64.csv'
# outfile = 'gpt_output_1024.csv'
f = open(outfile, "w")
f.write("ID,Label,Prediction,Result\n")

model = 'gpt-3.5-turbo-0613'
# model = 'ft:gpt-3.5-turbo-0613:personal:yelp-64-3:8ByzNBK8'
# model = 'ft:gpt-3.5-turbo-0613:personal:yelp-1024-3:88nshQ8D'

for i, input in enumerate(inputs):
  print(i + 1, end=" ")
  try:
    output=openai.ChatCompletion.create(
      model=model,
      messages=input["messages"][0:2],
      temperature=0
    ).choices[0].message.content
    print(output)
    f.write(f"{(i + 1)},{input['messages'][2]['content']},{output},{int(output == input['messages'][2]['content'])}\n")
    f.flush()
  except Exception as err:
    print(err)
  finally:
    sleep(0.1)

f.close()