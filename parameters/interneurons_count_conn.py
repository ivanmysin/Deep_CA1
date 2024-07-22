import pandas as pd

interneurons = pd.read_excel("/home/ivan/Data/Large_scale_CA1/interneurons.xlsx", sheet_name="Sheet2")

with open("presyns.txt", "r") as pfile:
    presyns = pfile.read()



with open("postsyns.txt", "r") as pfile:
    postsyns = pfile.read()

for cell_type in interneurons["Interneurons"]:
    cell_type = str(cell_type)

    count_presyn = presyns.count(cell_type)
    count_postsyn = postsyns.count(cell_type)

    print(cell_type, count_presyn, count_postsyn)