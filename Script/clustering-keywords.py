import cohere 
import matplotlib.pyplot as plt


co = cohere.Client('XXXXXXXXXXXXXXXXXXXX')
words = ['marketing','digital','seo','serp','datascience','python','new york','crypto','bitcoin','ethereum'] 
response = co.embed(model='large', texts=words) 
embeddings = response.embeddings

x = [embed[0] for embed in embeddings] 
y = [embed[1] for embed in embeddings] 


plt.figure(figsize=(20,10)) 
plt.scatter(x, y, s=500, c='green') 
for i, word in enumerate(words): 
    plt.annotate(word, (x[i], y[i]), fontsize=20) 
plt.show() 
