import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ps = [ 0.02371448 , 0.02802985,0.03178398 , 0.04927271 , 0.36864296]


data = [[0.01252344623208046, 0.00432620570063591, 0.0031896978616714478, 0.005202345550060272, 0.01846248283982277,0.023545119911432266, 0.02390214614570141, 0.006256180349737406], [0.07211681455373764, 0.10972553491592407, 0.02401588298380375, 0.04580869898200035, 0.00528377341106534, 0.04832177236676216, 0.008440978825092316, 0.005920870695263147], [0.040368691086769104, 0.05082332342863083, 0.01720399782061577, 0.010314250364899635, 0.031088000163435936, 0.0188534427434206, 0.011239136569201946, 0.021255064755678177], [0.07454148679971695, 0.06588638573884964, 0.009232209995388985, 0.007412655744701624, 0.03148883581161499, 0.04964553937315941, 0.027514398097991943, 0.10569407045841217], [0.03372815251350403, 0.02750188298523426, 0.011753376573324203, 0.007821399718523026, 0.007590844761580229, 0.018584176898002625, 0.03247309476137161, 0.07739132642745972]]


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


WordsInSent = [[u'city', u'qaeda', u'affiliate', u'rebels', u'rebranded', u'besieged', u'al', u'sham', u'fateh', u'jabhat'], [u'sending', u'-', u'tanks', u'syria', u'why', u'turkey', u'al', u'assad', u'?', u'bashar'], [u'so', u'of', u'deal', u'the', u'failed', u'the', u'price', u'remains', u'conflict', u'intractable'], [u'civil', u'beyond', u'intervention', u'protests', u'forceful', u'2011', u'in', u'international', u'war', u'syria'], [u'forward', u"'s", u'fast', u'very', u'certainly', u'looks', u'prognosis', u'diplomatic', u'bare', u'cupboard']]
head = [u'syria', u'deadlock', u'why', u'ca', u'vb']
WordsInSent = np.asarray(WordsInSent)
head = np.asarray(head)



words = []
for i in range(5):
    words.append(( WordsInSent[i,:8].tolist()))

words = np.asarray(words)
print (words)
print (words.shape[0])
print (words.shape[1])


print df[['c1', 'c2', 'c3','c4','c5', 'c6', 'c7', 'c8']].shape
print data.shape

ax = sns.clustermap(df[['c1', 'c2', 'c3','c4','c5', 'c6', 'c7', 'c8']], row_colors=network_colors, cmap=plt.cm.Reds, col_cluster=False, row_cluster=False)


plt.show()
