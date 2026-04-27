import pickle
import pandas as pd
import spacy

# 1. Carga de componentes
print("Cargando Motor Híbrido...")

# Carga de la MATRIZ de IBM 1 (ya no es un modelo NLTK, es un diccionario)
with open("../model/Model_IBM1.pkl", "rb") as f:
    model_dict = pickle.load(f)

# Cargar spacy y glosario
nlp = spacy.load("en_core_web_sm")
df = pd.read_csv("../data/banco_terminos_paso_B_para_traducir.csv")
medicDictionary = dict(zip(df.Termino_Ingles.str.lower(), df.Termino_Espanol))

# Inyectar reglas a SpaCy
ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = [{"label": "MED", "pattern": row['Termino_Ingles'].lower()} for _, row in df.iterrows()]
ruler.add_patterns(patterns)


# 2. Lógica de traducción
def IBMTranslation(tokens_en):
    translation = []
    for word in tokens_en:
        if word.lower().startswith("[term_"):
            translation.append(word)
            continue

        probabilities = model_dict.get(word, None)

        if probabilities:
            bestWord = max(probabilities, key=probabilities.get)
            translation.append(bestWord if bestWord else word)
        else:
            translation.append(word)

    return translation


def finalTranslation(enText):
    # A. Enmascaramiento
    doc = nlp(enText.lower())
    sentence = enText.lower()
    replacementMap = {}

    for i, ent in enumerate(doc.ents):
        if ent.label_ == "MED":
            label = f"[term_{i}]"  # Forzamos minúsculas para evitar cruces
            replacementMap[label] = medicDictionary[ent.text]
            sentence = sentence.replace(ent.text, label)

    sentence = sentence.replace(".", " .").replace(",", " ,").replace(":", " :")

    # B. Traducir con estadística
    tokens_en = sentence.split()
    tokens_es_prel = IBMTranslation(tokens_en)

    # C. Desenmascaramiento (Inserción de reglas)
    finalResult = []
    for token in tokens_es_prel:
        if token.lower() in replacementMap:
            finalResult.append(replacementMap[token.lower()])
        else:
            finalResult.append(token)

    # D. Limpieza final
    resultado_texto = " ".join(finalResult).capitalize()
    resultado_texto = resultado_texto.replace(" .", ".").replace(" ,", ",").replace(" :", ":")

    return resultado_texto


if __name__ == '__main__':
    # Prueba de escritorio múltiple
    frases_prueba = [
        "The doctor detected new malignancies near the breast cancer.",
        "A primary breast sarcoma is a rare type of cancer.",
        "Treatments for primary breast sarcomas are very complex."
    ]

    print("\n--- INICIANDO BATERÍA DE PRUEBAS ---")
    for frase in frases_prueba:
        resultado = finalTranslation(frase)
        print(f"\nEN: {frase}")
        print(f"ES: {resultado}")