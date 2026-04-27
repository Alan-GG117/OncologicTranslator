import pandas as pd
import spacy
from spacy.pipeline import EntityRuler

print("Iniciando modelo NER...")

glosaryPath = "../data/banco_terminos_paso_B_para_traducir.csv"
df = pd.read_csv(glosaryPath)

medicDictionary = dict(zip(df.Termino_Ingles, df.Termino_Espanol))
print(f"Glosario cargado con {len(medicDictionary)} terminos oncologicos.")

nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")

pattern = []
for termino_en, tipo in zip(df.Termino_Ingles, df.Tipo_Entidad):
    pattern.append({"label": tipo, "pattern": str(termino_en)})

ruler.add_patterns(pattern)
print("Motor spacy configurado con reglas oncologicas.")

def enmascararOracion(oracion_ingles):
    doc = nlp(oracion_ingles.lower())

    oracion_enmascarada = oracion_ingles.lower()
    terminos_encontrados = {}
    contador = 0

    # Buscar las entidades (términos) que definimos en las reglas
    for ent in doc.ents:
        if ent.label_ in ['DISEASE', 'CHEMICAL']:
            etiqueta_mascara = f"[TERMINO_MEDICO_{contador}]"
            terminos_encontrados[etiqueta_mascara] = medicDictionary[ent.text]
            oracion_enmascarada = oracion_enmascarada.replace(ent.text, etiqueta_mascara)
            contador += 1

    return oracion_enmascarada, terminos_encontrados

oracion_prueba = "A new treatment for breast cancer and primary breast sarcoma was discovered."
mascara, diccionario_reemplazos = enmascararOracion(oracion_prueba)

print("\n--- RESULTADOS DE LA PRUEBA ---")
print(f"Oración Original: {oracion_prueba}")
print(f"Oración Enmascarada: {mascara}")
print(f"Traducciones protegidas a insertar después: {diccionario_reemplazos}")