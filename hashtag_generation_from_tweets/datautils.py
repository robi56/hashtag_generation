import codecs
import json
import re
#load twitter data
statusText='statusText'
hashTagEntities='hashtagEntities'


#preprocessing

def removeTagUrlContent(text):
    text = text.replace("#","")
    text = re.sub(r"http\S+", "", text)
    return str(text)

#load twitter data
def loadTwitterData(dataUrl):
    data=[]
    with codecs.open(dataUrl,'rf') as f:
         lines = f.read().split('\n')
    contents, labels=[],[]
    for line in lines:
      #  print line
        status,hashtags="",[]
        try:
            tweet = json.loads(line)
            if statusText in tweet:
                status = tweet['statusText']
            else:
                continue
            if hashTagEntities in tweet:
                hashtagsinfo = tweet['hashtagEntities']
                for tagentity in hashtagsinfo:
                    tag = tagentity['text']
                    hashtags.append(tag)
                    #print tag

            else:
                continue
            contents.append(removeTagUrlContent(status))
            labels.append(hashtags)
            #print hashtags
        except Exception as exception:
            print exception
    return [contents, labels]



def buildingDictionaryForOutputLabels(labels):
    dict=[]
    count =1
    for labelset in labels:
        for label in labelset:
            if label in dict:
               count = count+1
            else:
                dict.append(label)
                count = count+1

    return dict



def numericalRepresentationOutputLabels(dic,labels):
    output=[]
    for labelset in labels:
        output_x=[]
        for label in labelset:
            test = lambda c: c == label
            index = map(test, dic).index(True)
            output_x.append(index)
        output.append(output_x)

    return output

