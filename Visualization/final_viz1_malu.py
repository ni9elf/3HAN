import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ps = [ 0.1198154 ,0.13790154,  0.14302,     0.14813468,  0.28681099]


#data = [[0.01252344623208046, 0.00432620570063591, 0.0031896978616714478, 0.005202345550060272, 0.01846248283982277,0.023545119911432266, 0.02390214614570141, 0.006256180349737406], [0.07211681455373764, 0.10972553491592407, 0.02401588298380375, 0.04580869898200035, 0.00528377341106534, 0.04832177236676216, 0.008440978825092316, 0.005920870695263147], [0.040368691086769104, 0.05082332342863083, 0.01720399782061577, 0.010314250364899635, 0.031088000163435936, 0.0188534427434206, 0.011239136569201946, 0.021255064755678177], [0.07454148679971695, 0.06588638573884964, 0.009232209995388985, 0.007412655744701624, 0.03148883581161499, 0.04964553937315941, 0.027514398097991943, 0.10569407045841217], [0.03372815251350403, 0.02750188298523426, 0.011753376573324203, 0.007821399718523026, 0.007590844761580229, 0.018584176898002625, 0.03247309476137161, 0.07739132642745972]]


data = [[[0.8636728525161743], [0.033481668680906296], [0.0009150446858257055], [0.0005172566743567586], [0.008213917724788189], [0.00900469534099102], [0.0019295376259833574], [0.0006996827432885766]], [[0.34459778666496277], [0.023884441703557968], [0.0018409626791253686], [0.00017506313452031463], [7.102506060618907e-05], [6.496557762147859e-05], [0.00309689249843359], [0.0028084460645914078]], [[0.0949239432811737], [0.0012599159963428974], [0.0007647867896594107], [0.00019136478658765554], [5.417149077402428e-05], [0.0001888442347990349], [2.876650250982493e-05], [2.8919190299347974e-05]], [[0.03870558738708496], [9.135083382716402e-05], [1.1767135219997726e-05], [4.768428425450111e-06], [1.301962856814498e-05], [1.1141621143906377e-05], [7.076033944031224e-05], [2.037716876657214e-05]], [[0.18027788400650024], [0.037016890943050385], [0.14164794981479645], [0.38200387358665466], [0.15511459112167358], [0.048632387071847916], [0.01737159490585327], [0.012263507582247257]]]

data = [[k[0] for k in line] for line in data]


data1 = []
i = 0
for e in ps:
    #e = e[0]
    e = e**(1/2.0)
    l = [a*e for a in data[i]]
    data1.append(l)
    i += 1
data = np.asarray(data)
data1 = np.asarray(data1)
#print data1
#print data

ps = np.asarray(ps)
        
d = {'c1':data1[:,0],
        'c2':data1[:,1 ],
        'c3':data1[:,2 ],
        'c4':data1[:,3 ],
        'c5':data1[:,4 ],
        'c6':data1[:,5 ],
        'c7':data1[:,6 ],
        'c8':data1[:,7 ],
        'si':ps}

for key , val in d.items():
    print key
    print val
    print               
df=pd.DataFrame(d)
#df[:3]

#--------------------------------------------------------------------------------------------
#create a color palette with the same number of colors as unique values in the Source column
network_pal = sns.light_palette('green', len(df.si.unique()))

#Create a dictionary where the key is the category and the values are the
#colors from the palette we just created
network_lut = dict(zip(df.si.unique(), network_pal))

#get the series of all of the categories
networks = df.si

#map the colors to the series. Now we have a list of colors the same
#length as our dataframe, where unique values are mapped to the same color
network_colors = pd.Series(networks).map(network_lut)

#plot the heatmap with the 16S and ITS categories with the network colors

#defined by Source column
#ax = sns.clustermap(df[['16S', 'ITS']], row_colors=network_colors, cmap='BuGn_r')

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------


#WordsInSent = [[u'city', u'qaeda', u'affiliate', u'rebels', u'rebranded', u'besieged', u'al', u'sham'], [u'sending', u'-', u'tanks', u'syria', u'why', u'turkey', u'al', u'assad'], [u'so', u'of', u'deal', u'the', u'failed', u'the', u'price', u'remains'], [u'civil', u'beyond', u'intervention', u'protests', u'forceful', u'2011', u'in', u'international'], [u'forward', u"'s", u'fast', u'very', u'certainly', u'looks', u'prognosis', u'diplomatic']]


WordsInSent = [[u'we', u'live', u'in', u'a', u'truly', u'orwellian', u'world', u'or', u',', u'perhaps', u',', u'it', u'is', u'more', u'of', u'an', u'aldous', u'huxley', u'dystopia'], [u'either', u'way', u',', u'when', u'a', u'government', u'finds', u'itself', u'ensnarled', u'by', u'the', u'"', u'controversial', u'"', u'act', u'of', u'providing', u'security', u'for', u'the', u'nation', u"'s", u'citizens', u',', u'all', u'reason', u'and', u'rationality', u'has', u'departed.', u',', u'president', u'trump', u"'s", u'idea', u'of', u'a', u'passive', u'method', u'of', u'stemming', u'illegal', u'immigration', u',', u'his', u'proposed', u'wall', u',', u'has', u'been', u'the', u'target', u'of', u'unprecedented', u'vilification'], [u'the', u'bill', u',', u'if', u'it', u'becomes', u'law', u',', u'could', u'net', u'up', u'to', u'130', u'billion', u'a', u'year', u',', u'no', u'small', u'chunk', u'of', u'change.', u',', u'republican', u'congressman', u'mike', u'rogers', u'of', u'alabama', u'is', u'quoted', u'as', u'saying', u',', u'"', u'this', u'bill', u'is', u'simple', u'-', u'-', u'anyone', u'who', u'sends', u'their', u'money', u'to', u'countries', u'that', u'benefit', u'from', u'our', u'porous', u'borders', u'and', u'illegal', u'immigration', u'should', u'be', u'responsible', u'for', u'providing', u'some', u'of', u'the', u'funds', u'needed', u'to', u'complete', u'the', u'wall', u'.', u'"'], [u',', u'before', u'any', u'liberal', u'reading', u'this', u'decides', u'to', u'rush', u'out', u'and', u'paint', u'a', u'"', u'trump', u'is', u'a', u'xenophobe', u'"', u'sign', u',', u'they', u'may', u'want', u'to', u'consider', u'that', u'the', u'proposed', u'2', u'percent', u'tax', u'is', u'rather', u'small', u'compared', u'to', u'the', u'international', u'average', u'of', u'8', u'percent'], [u'even', u'refugee', u'welcoming', u'canada', u'levies', u'a', u'12', u'percent', u'penalty', u'on', u'immigrant', u'money']]

head = [u'syria', u'deadlock', u'why', u'ca', u'vb']
WordsInSent = np.asarray(WordsInSent)
head = np.asarray(head)



words = []
for i in range(5):
    words.append(( WordsInSent[i][:8]))

words = np.asarray(words)
print (words)
print (words.shape[0])
print (words.shape[1])


print df[['c1', 'c2', 'c3','c4','c5', 'c6', 'c7', 'c8']].shape
print data.shape

#ax = sns.clustermap(df[['c1', 'c2', 'c3','c4','c5', 'c6', 'c7', 'c8']], row_colors=network_colors, cmap=plt.cm.Reds, col_cluster=False, row_cluster=False)

#Changing row names of dataframe df to weights in ps

df.index = [round(x,3) for x in ps]
#Annot neads a matrix of values for annotation of each cell, linewidth gives a small gap between each cell
#Note there is a bug in seaborn heatmap, cannot produce annotation when xticklabels are FALSE, workaround create an array of empty strings to pass as xticklabels. https://github.com/mwaskom/seaborn/issues/837
plt.figure(figsize=(12, 4))
new_x_labels = [""]*df[['c1', 'c2', 'c3','c4','c5', 'c6', 'c7', 'c8']].shape[1]
ax = sns.heatmap(df[['c1', 'c2', 'c3','c4','c5', 'c6', 'c7', 'c8']],annot=words,fmt ="",cmap=plt.cm.Reds,linewidths=.5,xticklabels=new_x_labels)

#plt.show()
plt.savefig('destination_path1.eps', format='eps', dpi=200)
plt.close()
