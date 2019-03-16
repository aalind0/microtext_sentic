import preprocessor as p
import re
another_iv = open("another_ovv.txt","r").read()

for i in another_iv.split('\n'):
	new = p.clean(i)
	new = re.sub(':','',new).encode('utf-8')
	print(new)
