
fr = open("FR.csv","r")
fr_examples = fr.readlines()

for example in fr_examples:
	print(example.strip()+",1")

en = open("EN.csv","r")
en_examples = en.readlines()

for example in en_examples:
       print(example.strip()+",0")



