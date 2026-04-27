import pandas as pd
import ast
import pickle
import time
import os
from nltk.translate import AlignedSent
from nltk.translate.ibm1 import IBMModel1

def trainingAndSaving():
    print("Iniciando entrenamiento de modelo IBM1")

    # 1. Definir rutas relativas
    dataPath = "../data/Corpus_Tokenizado_Final.csv"
    modelPath = "../model/Model_IBM1.pkl"

    # Verificación de la existencia del modelo
    os.makedirs(os.path.dirname(modelPath), exist_ok=True)

    # 2. Cargar dataset
    print(f"Cargando corpus tokenizado desde {dataPath}")
    try:
        df = pd.read_csv(dataPath)
    except FileNotFoundError:
        print("ERROR: El archivo no existe")
        return

    # 3. Deserializar los tokens
    print("Procesando y formateando tokens...")
    df['Tokens_Ingles'] = df['Tokens_Ingles'].apply(ast.literal_eval)
    df['Tokens_Espanol'] = df['Tokens_Espanol'].apply(ast.literal_eval)

    # 4. Creación de las alineaciones bi-textuales
    print("Generando matriz de oraciones alineadas...")
    trainingCorpus = []
    for es, en in zip(df['Tokens_Espanol'], df['Tokens_Ingles']):
        trainingCorpus.append(AlignedSent(es, en))

    print(f" -> Total de oraciones listas para iterar: {len(trainingCorpus)}")

    # 5. Entrenamiento del modelo (Expectation-Maximization)
    iterations = 10
    print(f"\nIniciado entrenamiento estadístico ({iterations} iteraciones)")
    start = time.time()
    model = IBMModel1(trainingCorpus, iterations)
    total = time.time() - start
    print(f"Entrenamiento finalizado en {total} segundos")

    # 6. Serializacion (Guardar el Modelo)
    print("\nGuardando el modelo... ")
    with open(modelPath, 'wb') as file:
        pickle.dump(model, file)

    print(f"Modelo guardado correctamente en: {modelPath}")
    print("\n--- PROCESO COMPLETADO ---")
    print("El microservicio FastAPI ya puede consumir este archivo.")

if __name__ == "__main__":
    trainingAndSaving()