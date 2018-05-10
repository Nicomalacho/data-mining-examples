from  sklearn import tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

# [Altura, Peso, Talla]
x = [[181,80,44], [177,70,43], [160,60,38], [154,54,37],
	[166,65,40], [190,90,47], [175,64,39], [177,70,40],
	[171,75,42], [181,85,43], [176,65,38], [182,98,41],
	[175,80,35], [160,65,39], [167,54,36], [184,88,42]]

y = ['hombre','hombre','mujer','mujer',
	'mujer','hombre','mujer','mujer',
	'mujer','hombre','mujer','hombre',
	'mujer','mujer','mujer','hombre']

clf = tree.DecisionTreeClassifier()

clf = clf.fit(x,y)


print ">>>>> Prediction <<<<<"

prediction = clf.predict([[183,88,42],[167,58,37],[172,75,39]])
print prediction

out = StringIO()
export_graphviz(clf, out_file=out,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(out.getvalue())  
Image(graph.create_png())
graph.write_pdf("tree.pdf")