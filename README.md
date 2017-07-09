# hashtag_generation
hashtag generation from  Social Media Data 
## Example 1: Hashtag generation from tweets collected bt tweet api
This is a multilabel categorical classification problem. But in the project , the input tweet is repeated for its corresponding hashtags and  has been made it a general general classification problem. The output is predicated by ranking consedering the hashtags are is its corrensponding tweet.  
 ### Collect Pretrained Word2Vector Model 
 Step 1: Download pretrained glob vector from  https://nlp.stanford.edu/projects/glove/
 Step 2: keep the file in glob/ folder 

### Train 
 python hashtagbycnn.py 
 
### Test 
python test_twitter_model.py



