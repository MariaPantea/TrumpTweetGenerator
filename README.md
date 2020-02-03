# TrumpTweetGenerator
This is a machine learning model which is trained on Trump's twitter data to genrate new tweets written like Trump would have expressed it. It's a small project made for a preseantation at Camp Vera 2020. Thew model is a simple RNN with word2vec embeddings. The data is collected at [trumptwitterarchive](http://www.trumptwitterarchive.com/archive) and postprocessed a bit to get cleaner data. 

Use however you like. 
Happy coding :) 

# How to run:
1. Download the Glove embeddings (glove.6B.zip) from [stanford](https://nlp.stanford.edu/projects/glove/) and add the 50d in the glove folder.
2. run `pip install -r requirements.txt` to install the packages
3. Run the program to use the pretrained model, optional flag `--start` and the start of the sentence to generate a specific topic. 
    ex. `python3 main.py --start the wall`
4. To re-train the model, add the flag `--train True`
