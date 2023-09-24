import os

import json


def writeNameLabelToFile(file):

	label = {'nnumber':"",
		  'prep':"",
		  'solid/inf':"",
		  'differentiation':"",
		  'idh-status':"",
		  'grade':0,
		  'quality':2
		}

	fileSplit = file.split(".")
	fileSplit = fileSplit[0].split("-")
	if len(fileSplit) ==6:
		if fileSplit[0].startswith("inf"):
			label["solid/inf"] = 'inf'
		else:
			label["solid/inf"] = 'solid'

		if fileSplit[1].startswith('O'):
			label['differentiation'] = 'oligodendroglial'
			label['idh-status'] = "mutated"
		elif fileSplit[1].startswith('GBM'):
			label['differentiation'] = 'astrocytic'
			label['idh-status'] = "wild"

		elif fileSplit[1].startswith('A'):
			label['differentiation'] = 'astrocytic'
			label['idh-status'] = "mutated"

		if fileSplit[1][-1].isdigit():
			label['grade'] = fileSplit[1][-1] 
		else:
			label['grade'] = 4

		nNumber = fileSplit[2]+"-"+fileSplit[3]
		label['nnumber'] = nNumber


		if fileSplit[4].startswith("K"):
			label['prep'] = "kryo"
		elif fileSplit[4].startswith("Q"):
			label['prep'] = "smear"
		if fileSplit[4].startswith("T"):
			label['prep'] = "touch"
		label['quality']=fileSplit[5][-1]
	else:
		if fileSplit[0].startswith("inf"):
			label["solid/inf"] = 'inf'
		else:
			label["solid/inf"] = 'solid'

		if fileSplit[0].startswith('O'):
			label['differentiation'] = 'oligodendroglial'
			label['idh-status'] = "mutated"
		elif fileSplit[0].startswith('GBM'):
			label['differentiation'] = 'astrocytic'
			label['idh-status'] = "wild"

		elif fileSplit[0].startswith('A'):
			label['differentiation'] = 'astrocytic'
			label['idh-status'] = "mutated"

		if fileSplit[0][-1].isdigit():
			label['grade'] = fileSplit[1][-1] 
		else:
			label['grade'] = 4

		nNumber = fileSplit[1]+"-"+fileSplit[2]
		label['nnumber'] = nNumber


		if fileSplit[3].startswith("K"):
			label['prep'] = "kryo"
		elif fileSplit[3].startswith("Q"):
			label['prep'] = "smear"
		if fileSplit[3].startswith("T"):
			label['prep'] = "touch"
		label['quality']=fileSplit[4][-1]

	return label


def writeJsonFile(dict,outPath):
	with open(outPath, "w") as outfile:
		json.dump(dict, outfile)

def main(inPath, outPath):

	for slide in os.listdir(inPath):
		print("---------------")
		label = writeNameLabelToFile(slide)

		print(label)
		print("---------------")

		slide = slide.replace(".svs", ".json")

		jsonPath = os.path.join(outPath,slide)

		writeJsonFile(label, jsonPath)



if __name__ == "__main__":

	inpath = r"C:\Users\felix\Desktop\neuro\kryoTest"
	outPath = r"C:\Users\felix\Desktop\neuro\kryoTest"

	main(inpath,outPath)





	



	

		





	