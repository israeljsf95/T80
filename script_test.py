from PIL import Image
import numpy as np
import os


image_dir = "C:/Users/Israel/Documents/UFS/Projeto de Astronomia/Tarefas/Fazer a classificacao/allsky_april/mnt/public/allsky_2"
os.chdir("C:/Users/Israel/Documents/UFS/Projeto de Astronomia/Tarefas/Fazer a classificacao/allsky_april/mnt/public/allsky_2")
images = []
text_images = open("C:/Users/Israel/Documents/UFS/Projeto de Astronomia/Tarefas/Fazer a classificacao/allsky_april/mnt/public/class.txt", 'w')

#Preparando os Dados

for image in os.listdir(image_dir):
    images.append(image)

""" 
O script esta lendo as imagens que eu salvei 
porem nao esta criando o arquivo referente
"""

for i in range(len(images) - 2):
			im = Image.open(images[i])
			im.show()
			class_image = input("Boa(1), Media(0.5) ou Ruim(0)?: ")
			text_images.write(images[i] + ":" + str(class_image) + "\n")

text_images.close()
