import os
import cv2
import tarfile
import requests
import numpy as np
import matplotlib.pyplot as plt

pathToDataset = 'Images'
url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'

## Módulo para preparar o Dataset
def prepareDataset(url): 
    nome = url.split('/')
    nome = nome[len(nome)-1]
    #
    response = requests.get(url, stream=True) # Faz a requisição
    total_length = response.headers.get('content-length') # Pega o tamanho do arquivo
    if response.status_code == requests.codes.OK: # Verifica se a requisição ocorreu bem
        # Realiza Download #
        progress = 0 # Registra a quantidade de informação baixada
        with open(nome, 'wb') as arquivo:
            for parte in response.iter_content(chunk_size=256): 
                arquivo.write(parte)
                progress = progress + len(parte)
                porcent = (progress/int(total_length))*100
                print(f'Estamos em {round(porcent,1)}% de {total_length[:3]} Mb',end='\r')
        print("Download finalizado com sucesso.")
        # Extrai o arquivo #
        try: 
            t = tarfile.open(nome)
            print('Realizando extração, aguarde.')
            t.extractall()
            print('Arquivo extraido com sucesso.')
            t.close()
        except:
            print('Falha na extração do arquivo. Tente fazer manualmente.')
    else:
        print('Não foi possível realizar o download.')
        response.raise_for_status()

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
