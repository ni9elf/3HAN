import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------------------------------------------------
#for att3 - headline - body vector


headwords = [u'syria', u'deadlock', u'why', u'ca', u"n't", u'us', u',', u'russia', u'vb']
headalphas = [[0.02324029, 0.02358708, 0.01725436, 0.02079922, 0.01629241, 0.0131849, 0.01159455, 0.01222034, 0.83630037]]


ps = np.asarray(headalphas)
headwords = np.asarray(headwords)
#cmap = sns.cubehelix_palette(light=1, as_cmap=True)
cmap = sns.cubehelix_palette(dark=0.8, light=0.9, as_cmap=True)
#vmin, vmax = -0.02, 0.09
ax = sns.heatmap(ps, cmap=cmap)#, center=(vmin + vmax) / 2., vmax=vmax)

for y in range(ps.shape[0]):
	for x in range(ps.shape[1]):
		plt.text(x + 0.5, y + 0.5, '%s' % (headwords[x]), horizontalalignment='center', verticalalignment='center',)


plt.show()


'''
#---------------------------------------------------------------------------------------------------------------------------------

sent= [[u'led', u'by', u'the', u'former', u'al', u'qaeda', u'affiliate', u',', u'now', u'rebranded', u'as', u'jabhat', u'fateh', u'al', u'-', u'sham', u',', u'the', u'rebels', u'had', u'punched', u'through', u'a', u'corridor', u'to', u'besieged', u'parts', u'of', u'the', u'city'], [u'why', u'turkey', u'sending', u'tanks', u'into', u'syria', u'is', u'significant', u'what', u'to', u'do', u'about', u'bashar', u'al', u'-', u'assad', u'?'], [u'the', u'price', u'of', u'a', u'failed', u'deal', u'so', u'the', u'conflict', u'remains', u'intractable'], [u'in', u'2011', u',', u'as', u'street', u'protests', u'in', u'syria', u'began', u',', u'dean', u'forecast', u'that', u'without', u'immediate', u'and', u'forceful', u'international', u'intervention', u',', u'the', u'civil', u'war', u'would', u'last', u'10', u'years', u'and', u'leave', u'a', u'million', u'people', u'dead', u',', u'sucking', u'in', u'powers', u'around', u'the', u'region', u'and', u'beyond'], [u'fast', u'forward', u'five', u'years', u'and', u'dean', u"'s", u'prognosis', u'still', u'holds', u'up', u'and', u'for', u'now', u',', u'certainly', u',', u'the', u'diplomatic', u'cupboard', u'looks', u'very', u'bare']]

si = [ 0.02371448 , 0.02802985 , 0.03178398 , 0.04927271 , 0.36864296]

wordalpha = [[[0.01252344623208046], [0.00432620570063591], [0.0031896978616714478], [0.005202345550060272], [0.01846248283982277], [0.023545119911432266], [0.02390214614570141], [0.006256180349737406]], [[0.07211681455373764], [0.10972553491592407], [0.02401588298380375], [0.04580869898200035], [0.00528377341106534], [0.04832177236676216], [0.008440978825092316], [0.005920870695263147]], [[0.040368691086769104], [0.05082332342863083], [0.01720399782061577], [0.010314250364899635], [0.031088000163435936], [0.0188534427434206], [0.011239136569201946], [0.021255064755678177]], [[0.07454148679971695], [0.06588638573884964], [0.009232209995388985], [0.007412655744701624], [0.03148883581161499], [0.04964553937315941], [0.027514398097991943], [0.10569407045841217]], [[0.03372815251350403], [0.02750188298523426], [0.011753376573324203], [0.007821399718523026], [0.007590844761580229], [0.018584176898002625], [0.03247309476137161], [0.07739132642745972]]]
'''
