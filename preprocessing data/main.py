import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

pathToDataset = 'images/Images'

## Módulo para criar o target name
def createTargetName(pathToDataset):
    pastas = os.listdir(pathToDataset)
    pastas.sort() # Para padronizar
    racas = []
    for i in range(len(pastas)):
        nomeracas = pastas[i].split('-')
        racas.append(nomeracas[1])
    # Cria a pasta, se não houver
    try:
        os.mkdir('target')
    except:
        pass
    # Escrevendo em um arquivo
    with open('target/targetNames.txt','w') as arquivo:
        for i in range(len(racas)-1):
            arquivo.write(str(racas[i])+'\n')
        arquivo.write(str(racas[len(racas)-1]))


## Módulo para criar target (númerico)
# Com esta função, cada imagem terá um número como rótulo
def createTarget(pathToDataset):
    pastas = os.listdir(pathToDataset)
    pastas.sort() # Para padronizar
    racas = []
    alvo = 0
    #
    for i in range(len(pastas)):
        arquivos = os.listdir(pathToDataset+'/'+pastas[i])
        for i in range (0, len(arquivos)):
            racas.append(alvo)
        alvo = alvo + 1
    # Cria a pasta, se não houver
    try:
        os.mkdir('target')
    except:
        pass
    # Escrevendo em um arquivo
    with open('target/target.txt','w') as arquivo:
        for i in range(len(racas)-1):
            arquivo.write(str(racas[i])+'\n')
        arquivo.write(str(racas[len(racas)-1]))


## Módulo para criar target (nome da raça)
# Com esta função, cada imagem terá o nome da raça como rótulo
def targetTESTE(pathToDataset):
    pastas = os.listdir(pathToDataset)
    pastas.sort() # Para padronizar
    racas = []
    # 
    for i in range(len(pastas)):
        arquivos = os.listdir(pathToDataset+'/'+pastas[i])
        nomeracas = pastas[i].split('-')
        for i in range (0, len(arquivos)):
            racas.append(nomeracas[1])
    # Cria a pasta, se não houver
    try:
        os.mkdir('target')
    except:
        pass
    # Escrevendo em um arquivo
    with open('target/targetTESTE.txt','w') as arquivo:
        for i in range(len(racas)-1):
            arquivo.write(str(racas[i])+'\n')
        arquivo.write(str(racas[len(racas)-1]))


## Módulo para ler o target (númerico)
def loadTarget(pathToDataset):
    target = []
    with open('target/target.txt','r') as arquivo:
        for linha in arquivo:
            target.append(int(linha))
    return np.array(target) # Para ser aceito pelo TensorFlow, necessário o np.array()


## Módulo para ler o target name
# Função para obter a raça correspondente ao alvo númerico
def loadTargetNames(pathToDataset):
    targetName = []
    with open('target/targetNames.txt','r') as arquivo:
        for linha in arquivo:
            targetName.append(linha.replace('\n',''))
    return targetName


## Módulo para retornar as fotos no formato indicado TensorFlow
def loadFeatures(pathToDataset):
    pastas = os.listdir(pathToDataset)
    pastas.sort() # Para padronizar
    features = []
    for i in range(len(pastas)): # Acessando as pasta
        arquivos = os.listdir(pathToDataset+'/'+pastas[i])
        for j in range (0, len(arquivos)): # Acessando os arquivos
            image = cv2.imread(pathToDataset+'/'+pastas[i]+'/'+arquivos[j]) # Transformando arquivo de foto em matriz numpy
            features.append(image)
    return np.array(features)
