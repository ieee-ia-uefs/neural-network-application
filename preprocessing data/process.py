import os
import cv2
import tarfile
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from google.colab.patches import cv2_imshow

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
        nomeracas = pastas[i].split('-', 1)
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
        for i in range (148):
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
        nomeracas = pastas[i].split('-', 1)
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
        for j in range (148): # Acessando os arquivos até o 148°, pois a menor quantidade de fotos presentes em agluma raça é 148
            image = cv2.imread(pathToDataset+'/'+pastas[i]+'/'+arquivos[j]) # Transformando arquivo de foto em matriz numpy
            resized_image = cv2.resize(image, (240, 240))
            features.append(resized_image)
    return np.array(features)


def agruparRacas(nomes, targets, imagensRacas, pathToDataset):
  """Agrupa as raças em um dicionário no qual o nome da raça é cada uma das chaves."""
  racas = {}
  
  for i in range(len(imagensRacas)):
    racaAtual = nomes[racasTarget[i]] # Nome da raça que está sendo agrupada
    
    if not racas.get(racaAtual, 0):
      racas[racaAtual] = [] # Cria uma lista como valor caso não exista.
    racas[racaAtual].append(imagensRacas[i]) # Adiciona a imagem à lista da chave.
  return racas


def separarTesteTreino(targets, dicioRacas):
  """Separa os dados de testes e de treinos, com 33 % das imagens de cada raça."""
  img_treino = []
  trgt_treino = []
  img_teste = []
  trgt_teste = []
  targetRaca = 0
  
  for raca in dicioRacas.values():
    qntdImagens = len(raca)
    targets = np.full( (qntdImagens, 1), targetRaca) # Array com os targets dessa raça.
    
    imagens_treino, imagens_teste, targets_treino, target_teste = train_test_split(raca, targets, test_size=0.33, random_state=42)
    
    img_treino += list(imagens_treino)
    trgt_treino += list(targets_treino)
    img_teste += list(imagens_teste)
    trgt_teste += list(target_teste)
    targetRaca += 1
  
  return img_treino, trgt_treino, img_teste, trgt_teste


nomes = loadTargetNames(pathToDataset)
racasTarget = loadTarget(pathToDataset)
imagens = loadFeatures(pathToDataset)

dicio = agruparRacas(nomes, racasTarget, imagens, pathToDataset)

imgns_treino, targets_treino, imgns_teste, targets_teste = separarTesteTreino(racasTarget, dicio)
