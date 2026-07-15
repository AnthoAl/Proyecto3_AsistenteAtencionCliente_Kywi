"""Asistente conversacional de Kywi basado exclusivamente en el CSV mejorado.

La recuperación de productos es determinista: primero filtra Tipo Producto,
después marca, presupuesto y uso. Los modelos no agregan nombres, precios,
existencias ni políticas a la respuesta comercial.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CATALOGO_PATH = BASE_DIR / "catalogo_kywi_mejorado.csv"
POLITICAS_PATH = BASE_DIR / "politicas_kywi.txt"

MODELOS = {
    "question_answering": "MMG/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es-finetuned-sqac",
    "zero_shot": "Recognai/bert-base-spanish-wwm-cased-xnli",
    "summarization": "google/flan-t5-small",
    "sentiment": "pysentimiento/robertuito-sentiment-analysis",
    "text_generation": "mrm8488/spanish-gpt2",
}

INTENCIONES = [
    "recomendación de producto",
    "comparación de productos",
    "precio y presupuesto",
    "características del producto",
    "disponibilidad de producto",
    "garantía",
    "cambio o devolución",
    "envío",
    "forma de pago",
    "reclamo",
    "otra consulta",
]

INTENCIONES_PRODUCTO = {
    "recomendación de producto",
    "comparación de productos",
    "precio y presupuesto",
    "características del producto",
    "disponibilidad de producto",
}

POLITICAS_POR_INTENCION = {
    "garantía": "GARANTÍA",
    "cambio o devolución": "CAMBIOS Y DEVOLUCIONES",
    "reclamo": "CAMBIOS Y DEVOLUCIONES",
    "envío": "ENTREGA A DOMICILIO",
    "forma de pago": "MEDIOS DE PAGO",
}

SIN_PREFERENCIA = {
    "cualquiera",
    "cualquier marca",
    "me da igual",
    "no importa",
    "no tengo preferencia",
    "sin preferencia",
}

SIN_LIMITE = {
    "cualquiera",
    "me da igual",
    "no importa",
    "no tengo limite",
    "sin limite",
    "sin presupuesto",
}

PALABRAS_RUIDO_TIPO = {
    "busco",
    "buscar",
    "compara",
    "comparar",
    "cual",
    "cuales",
    "dame",
    "disponible",
    "disponibles",
    "economica",
    "economicas",
    "economico",
    "economicos",
    "el",
    "ella",
    "en",
    "la",
    "las",
    "los",
    "marca",
    "me",
    "muestra",
    "muestrame",
    "necesito",
    "opcion",
    "opciones",
    "para",
    "por",
    "producto",
    "productos",
    "que",
    "quiero",
    "recomienda",
    "recomendame",
    "tiene",
    "tienen",
    "una",
    "uno",
    "unos",
    "uso",
}

EJEMPLOS = [
    "Compara dos hidrolavadoras económicas.",
    "¿Qué taladros tienen?",
    "Busco una aspiradora para uso doméstico.",
    "Recomiéndame una manguera para jardín.",
    "¿Qué formas de pago acepta Kywi?",
    "Estoy molesto porque mi producto llegó dañado.",
]


@dataclass
class EstadoModelos:
    question_answering: Any
    zero_shot: Any
    summarization: Any
    sentiment: Any
    text_generation: Any


catalogo: pd.DataFrame | None = None
politicas = ""
vectorizador_productos: TfidfVectorizer | None = None
matriz_productos: Any = None
vectorizador_tipos: TfidfVectorizer | None = None
matriz_tipos: Any = None
tabla_tipos: pd.DataFrame | None = None
modelos: EstadoModelos | None = None


def normalizar(texto: str) -> str:
    texto = str(texto or "").lower().strip()
    texto = "".join(
        c
        for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def cargar_catalogo() -> None:
    global catalogo, politicas
    global vectorizador_productos, matriz_productos
    global vectorizador_tipos, matriz_tipos, tabla_tipos

    if not CATALOGO_PATH.exists():
        raise FileNotFoundError(f"No se encontró {CATALOGO_PATH.name} junto a la aplicación.")
    if not POLITICAS_PATH.exists():
        raise FileNotFoundError(f"No se encontró {POLITICAS_PATH.name} junto a la aplicación.")

    catalogo = pd.read_csv(
        CATALOGO_PATH,
        sep=";",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )
    requeridas = {
        "Nombre Producto",
        "Marca",
        "Categoría Normalizada",
        "Subcategoría",
        "Tipo Producto",
        "Es Producto Principal",
        "Precio USD",
        "Texto Búsqueda",
        "Enlace Web",
    }
    faltantes = requeridas - set(catalogo.columns)
    if faltantes:
        raise ValueError(f"El CSV mejorado no contiene: {sorted(faltantes)}")

    catalogo["precio_num"] = pd.to_numeric(catalogo["Precio USD"], errors="coerce")
    catalogo["tipo_norm"] = catalogo["Tipo Producto"].map(normalizar)
    catalogo["marca_norm"] = catalogo["Marca"].map(normalizar)
    catalogo["principal"] = catalogo["Es Producto Principal"].map(normalizar).eq("si")

    corpus = catalogo.apply(
        lambda fila: " ".join(
            [
                (normalizar(fila["Tipo Producto"]) + " ") * 5,
                (normalizar(fila["Nombre Producto"]) + " ") * 3,
                (normalizar(fila["Subcategoría"]) + " ") * 2,
                normalizar(fila["Marca"]),
                normalizar(fila["Categoría Normalizada"]),
                normalizar(fila["Texto Búsqueda"]),
            ]
        ),
        axis=1,
    ).tolist()
    vectorizador_productos = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matriz_productos = vectorizador_productos.fit_transform(corpus)

    tipos = []
    for tipo, grupo in catalogo.groupby("Tipo Producto", sort=True):
        ejemplos = " ".join(grupo["Nombre Producto"].head(4).astype(str))
        subcategorias = " ".join(grupo["Subcategoría"].drop_duplicates().head(4))
        tipos.append(
            {
                "tipo": tipo,
                "tipo_norm": normalizar(tipo),
                "documento": " ".join(
                    [(normalizar(tipo) + " ") * 6, normalizar(subcategorias), normalizar(ejemplos)]
                ),
                "cantidad": len(grupo),
            }
        )
    tabla_tipos = pd.DataFrame(tipos)
    vectorizador_tipos = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1
    )
    matriz_tipos = vectorizador_tipos.fit_transform(tabla_tipos["documento"])
    politicas = POLITICAS_PATH.read_text(encoding="utf-8")


def iniciar_modelos() -> EstadoModelos:
    global modelos
    if modelos is not None:
        return modelos

    dispositivo = 0 if torch.cuda.is_available() else -1
    print(f"Cargando modelos en {'GPU' if dispositivo == 0 else 'CPU'}...")

    def crear(tarea: str, identificador: str, clase_modelo: Any) -> Any:
        print(f"  - {tarea}: {identificador}")
        tokenizer = AutoTokenizer.from_pretrained(identificador)
        parametros = {
            "low_cpu_mem_usage": False,
            "device_map": None,
        }
        try:
            modelo = clase_modelo.from_pretrained(
                identificador, use_safetensors=True, **parametros
            )
        except Exception:
            modelo = clase_modelo.from_pretrained(
                identificador, use_safetensors=False, **parametros
            )
        tensores_meta = [
            nombre
            for nombre, parametro in modelo.named_parameters()
            if parametro.device.type == "meta"
        ]
        if tensores_meta:
            raise RuntimeError(
                f"{identificador} conservó tensores meta: {tensores_meta[:3]}"
            )
        return pipeline(
            tarea,
            model=modelo,
            tokenizer=tokenizer,
            device=dispositivo,
        )

    modelos = EstadoModelos(
        question_answering=crear(
            "question-answering", MODELOS["question_answering"], AutoModelForQuestionAnswering
        ),
        zero_shot=crear(
            "zero-shot-classification", MODELOS["zero_shot"], AutoModelForSequenceClassification
        ),
        summarization=crear(
            "summarization", MODELOS["summarization"], AutoModelForSeq2SeqLM
        ),
        sentiment=crear(
            "sentiment-analysis", MODELOS["sentiment"], AutoModelForSequenceClassification
        ),
        text_generation=crear(
            "text-generation", MODELOS["text_generation"], AutoModelForCausalLM
        ),
    )
    return modelos


def interpretar_sentimiento(resultado: dict[str, Any]) -> tuple[str, float, bool]:
    etiqueta = str(resultado.get("label", "NEU")).upper()
    confianza = float(resultado.get("score", 0.0))
    mapa = {
        "NEG": "negativo",
        "NEU": "neutral",
        "POS": "positivo",
        "LABEL_0": "negativo",
        "LABEL_1": "neutral",
        "LABEL_2": "positivo",
    }
    sentimiento = mapa.get(etiqueta, etiqueta.lower())
    return sentimiento, confianza, sentimiento == "negativo" and confianza >= 0.60


def detectar_intencion(pregunta: str) -> tuple[str, float]:
    assert modelos is not None
    consulta = normalizar(pregunta)
    reglas = [
        ("comparación de productos", ["compara", "comparar", "diferencia entre"]),
        ("cambio o devolución", ["devolver", "devolucion", "cambiar producto"]),
        ("garantía", ["garantia"]),
        ("forma de pago", ["forma de pago", "formas de pago", "tarjeta", "pagar"]),
        ("envío", ["envio", "entrega", "domicilio"]),
        ("reclamo", ["reclamo", "danado", "defectuoso", "molesto"]),
        ("recomendación de producto", ["recomienda", "recomendame", "me conviene"]),
        ("precio y presupuesto", ["precio", "cuesta", "presupuesto"]),
    ]
    for intencion, indicadores in reglas:
        if any(indicador in consulta for indicador in indicadores):
            return intencion, 0.99
    if (
        any(palabra in consulta for palabra in ["que", "cuales", "muestra", "tienen"])
        and any(palabra in consulta for palabra in ["tiene", "tienen", "hay", "muestra"])
    ):
        return "disponibilidad de producto", 0.99

    resultado = modelos.zero_shot(
        pregunta,
        candidate_labels=INTENCIONES,
        hypothesis_template="Esta consulta trata sobre {}.",
    )
    return str(resultado["labels"][0]), float(resultado["scores"][0])


def limpiar_para_tipo(texto: str) -> str:
    consulta = normalizar(texto)
    if catalogo is not None:
        marcas = sorted(
            set(catalogo["marca_norm"]), key=len, reverse=True
        )
        for marca in marcas:
            if marca:
                consulta = re.sub(rf"\b{re.escape(marca)}\b", " ", consulta)
    palabras = [
        palabra
        for palabra in consulta.split()
        if palabra not in PALABRAS_RUIDO_TIPO and not palabra.isdigit()
    ]
    return " ".join(palabras)


def variantes_tipo(tipo: str) -> set[str]:
    normalizado = normalizar(tipo)
    variantes = {normalizado}
    if " " not in normalizado:
        if normalizado.endswith("s"):
            variantes.add(normalizado[:-1])
        else:
            variantes.add(normalizado + "s")
            if normalizado.endswith("or"):
                variantes.add(normalizado + "es")
    if normalizado == "hidrolavadora":
        variantes.add("hidro lavadora")
    return {variante for variante in variantes if variante}


def detectar_tipo_producto(texto: str) -> tuple[str | None, float]:
    assert tabla_tipos is not None and vectorizador_tipos is not None
    consulta = limpiar_para_tipo(texto)
    if not consulta:
        return None, 0.0

    coincidencias = []
    for _, fila in tabla_tipos.iterrows():
        for variante in variantes_tipo(str(fila["tipo"])):
            if re.search(rf"\b{re.escape(variante)}\b", consulta):
                coincidencias.append((len(variante), str(fila["tipo"])))
    if coincidencias:
        coincidencias.sort(reverse=True)
        return coincidencias[0][1], 1.0

    similitudes = cosine_similarity(
        vectorizador_tipos.transform([consulta]), matriz_tipos
    )[0]
    posicion = int(similitudes.argmax())
    confianza = float(similitudes[posicion])
    if confianza < 0.22:
        return None, confianza
    return str(tabla_tipos.iloc[posicion]["tipo"]), confianza


def candidatos_del_tipo(tipo: str | None) -> pd.DataFrame:
    assert catalogo is not None
    if not tipo:
        return catalogo.iloc[0:0].copy()
    candidatos = catalogo[catalogo["tipo_norm"].eq(normalizar(tipo))].copy()
    principales = candidatos[candidatos["principal"]]
    return principales if not principales.empty else candidatos


def marcas_disponibles(tipo: str | None, limite: int = 8) -> list[str]:
    candidatos = candidatos_del_tipo(tipo)
    if candidatos.empty:
        return []
    marcas = candidatos["Marca"].astype(str).str.strip()
    marcas = marcas[marcas.ne("")]
    return marcas.value_counts().head(limite).index.tolist()


def detectar_marca_global(texto: str) -> str | None:
    assert catalogo is not None
    consulta = normalizar(texto)
    marcas = sorted(set(catalogo["Marca"]), key=lambda valor: len(normalizar(valor)), reverse=True)
    for marca in marcas:
        marca_norm = normalizar(marca)
        if marca_norm and re.search(rf"\b{re.escape(marca_norm)}\b", consulta):
            return marca
    return None


def detectar_marca_del_tipo(texto: str, tipo: str | None) -> str | None:
    consulta = normalizar(texto)
    for marca in sorted(
        marcas_disponibles(tipo, limite=100),
        key=lambda valor: len(normalizar(valor)),
        reverse=True,
    ):
        marca_norm = normalizar(marca)
        if re.search(rf"\b{re.escape(marca_norm)}\b", consulta):
            return marca
    return None


def detectar_presupuesto(texto: str) -> float | None:
    texto_original = str(texto or "").lower().replace(",", ".")
    con_simbolo = re.search(r"\$\s*(\d+(?:\.\d+)?)", texto_original)
    if con_simbolo:
        return float(con_simbolo.group(1))
    consulta = normalizar(texto).replace(",", ".")
    patrones = [
        r"(?:maximo|hasta|menos de|presupuesto de|no mas de)\s*(?:usd)?\s*(\d+(?:\.\d+)?)",
        r"(?:usd)\s*(\d+(?:\.\d+)?)",
    ]
    for patron in patrones:
        coincidencia = re.search(patron, consulta)
        if coincidencia:
            return float(coincidencia.group(1))
    coincidencia = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*", str(texto or ""))
    return float(coincidencia.group(1).replace(",", ".")) if coincidencia else None


def detectar_uso(texto: str) -> str | None:
    consulta = normalizar(texto)
    usos = {
        "uso profesional": ["profesional", "trabajo diario", "intensivo"],
        "uso doméstico": ["domestico", "hogar", "casa"],
        "uso ocasional": ["ocasional", "de vez en cuando", "pocas veces"],
        "uso industrial": ["industrial", "industria"],
    }
    for uso, indicadores in usos.items():
        if any(indicador in consulta for indicador in indicadores):
            return uso
    return None


def contiene_respuesta(texto: str, opciones: set[str]) -> bool:
    consulta = normalizar(texto)
    return any(normalizar(opcion) in consulta for opcion in opciones)


def estado_nuevo() -> dict[str, Any]:
    return {
        "pregunta_inicial": "",
        "intencion": "",
        "confianza_intencion": 0.0,
        "sentimiento": "",
        "confianza_sentimiento": 0.0,
        "tipo_producto": None,
        "confianza_tipo": 0.0,
        "marca": None,
        "marca_definida": False,
        "presupuesto": None,
        "presupuesto_definido": False,
        "uso": None,
        "pregunta_pendiente": None,
        "finalizado": False,
    }


def siguiente_pregunta(estado: dict[str, Any]) -> str | None:
    if not estado.get("tipo_producto"):
        estado["pregunta_pendiente"] = "tipo"
        ejemplos = [
            tipo
            for tipo in ["Taladro", "Hidrolavadora", "Aspiradora", "Manguera", "Cafetera"]
            if catalogo is not None and normalizar(tipo) in set(catalogo["tipo_norm"])
        ]
        return "¿Qué tipo de producto buscas?" + (
            f" Por ejemplo: {', '.join(ejemplos)}." if ejemplos else ""
        )

    if not estado.get("marca_definida"):
        estado["pregunta_pendiente"] = "marca"
        marcas = marcas_disponibles(estado.get("tipo_producto"))
        opciones = f" Las marcas disponibles son: {', '.join(marcas)}." if marcas else ""
        return (
            f"¿Qué marca prefieres para **{estado['tipo_producto']}**?"
            f"{opciones} También puedes responder **sin preferencia**."
        )

    if not estado.get("presupuesto_definido"):
        estado["pregunta_pendiente"] = "presupuesto"
        return "¿Cuál es tu presupuesto máximo? Escribe un valor o responde **sin límite**."

    if estado.get("intencion") in {
        "recomendación de producto",
        "comparación de productos",
    } and not estado.get("uso"):
        estado["pregunta_pendiente"] = "uso"
        return "¿Para qué lo utilizarás: uso doméstico, ocasional, profesional o industrial?"

    estado["pregunta_pendiente"] = None
    return None


def registrar_seguimiento(mensaje: str, estado: dict[str, Any]) -> tuple[bool, str]:
    pendiente = estado.get("pregunta_pendiente")
    if pendiente == "tipo":
        tipo, confianza = detectar_tipo_producto(mensaje)
        if not tipo:
            return False, "No pude identificar ese tipo de producto en el CSV. Escríbelo de otra manera."
        estado["tipo_producto"] = tipo
        estado["confianza_tipo"] = confianza
        return True, f"Producto identificado: **{tipo}**."

    if pendiente == "marca":
        if contiene_respuesta(mensaje, SIN_PREFERENCIA):
            estado["marca"] = None
            estado["marca_definida"] = True
            return True, "Buscaré entre todas las marcas disponibles para ese producto."
        marca = detectar_marca_del_tipo(mensaje, estado.get("tipo_producto"))
        if not marca:
            solicitada = detectar_marca_global(mensaje) or mensaje.strip()
            disponibles = marcas_disponibles(estado.get("tipo_producto"))
            return (
                False,
                f"No encontré **{estado['tipo_producto']}** de marca **{solicitada}**. "
                f"Las marcas disponibles son: {', '.join(disponibles)}. "
                "Elige una o responde **sin preferencia**.",
            )
        estado["marca"] = marca
        estado["marca_definida"] = True
        return True, f"Marca registrada: **{marca}**."

    if pendiente == "presupuesto":
        if contiene_respuesta(mensaje, SIN_LIMITE):
            estado["presupuesto"] = None
            estado["presupuesto_definido"] = True
            return True, "No aplicaré un límite de precio."
        presupuesto = detectar_presupuesto(mensaje)
        if presupuesto is None or presupuesto <= 0:
            return False, "Escribe un valor como **100** o responde **sin límite**."
        estado["presupuesto"] = presupuesto
        estado["presupuesto_definido"] = True
        return True, f"Presupuesto máximo: **${presupuesto:.2f}**."

    if pendiente == "uso":
        estado["uso"] = detectar_uso(mensaje) or mensaje.strip()
        return True, f"Uso registrado: **{estado['uso']}**."

    return False, "Pulsa **Nueva conversación** y escribe una consulta inicial."


def extraer_politica(titulo: str) -> str:
    patron = rf"{re.escape(titulo)}\n(.*?)(?=\n[A-ZÁÉÍÓÚÑ ]+\n|\nFUENTES OFICIALES|\Z)"
    coincidencia = re.search(patron, politicas, flags=re.DOTALL)
    return coincidencia.group(1).strip() if coincidencia else "No encontré esa sección en las políticas cargadas."


def buscar_segun_estado(estado: dict[str, Any]) -> tuple[pd.DataFrame, float]:
    assert vectorizador_productos is not None
    candidatos = candidatos_del_tipo(estado.get("tipo_producto"))
    if estado.get("marca"):
        candidatos = candidatos[candidatos["marca_norm"].eq(normalizar(estado["marca"]))]
    if estado.get("presupuesto") is not None:
        candidatos = candidatos[
            candidatos["precio_num"].le(float(estado["presupuesto"]))
        ]
    candidatos = candidatos[candidatos["precio_num"].notna()].copy()
    if candidatos.empty:
        return candidatos, 0.0

    consulta = " ".join(
        [estado.get("pregunta_inicial", ""), estado.get("uso", "") or ""]
    )
    similitudes = cosine_similarity(
        vectorizador_productos.transform([normalizar(consulta)]), matriz_productos
    )[0]
    candidatos["similitud"] = similitudes[candidatos.index]
    economico = any(
        palabra in normalizar(estado.get("pregunta_inicial", ""))
        for palabra in ["economico", "economica", "barato", "barata"]
    )
    if economico or estado.get("intencion") == "precio y presupuesto":
        candidatos = candidatos.sort_values(
            ["precio_num", "similitud"], ascending=[True, False]
        )
    else:
        candidatos = candidatos.sort_values(
            ["similitud", "precio_num"], ascending=[False, True]
        )
    return candidatos, float(candidatos["similitud"].max())


def tabla_salida(productos: pd.DataFrame) -> pd.DataFrame:
    columnas = ["Producto", "Marca", "Tipo", "Precio USD", "Disponibilidad", "Enlace"]
    if productos.empty:
        return pd.DataFrame(columns=columnas)
    salida = productos[
        [
            "Nombre Producto",
            "Marca",
            "Tipo Producto",
            "precio_num",
            "Disponibilidad",
            "Enlace Web",
        ]
    ].copy()
    salida.columns = columnas
    salida["Precio USD"] = salida["Precio USD"].map(lambda valor: f"${valor:.2f}")
    return salida


def alternativas_por_marca(tipo: str | None, minimo: int = 2) -> list[str]:
    candidatos = candidatos_del_tipo(tipo)
    conteos = candidatos.groupby("Marca").size().sort_values(ascending=False)
    return conteos[conteos >= minimo].head(6).index.tolist()


def ejecutar_auxiliares(pregunta: str, productos: pd.DataFrame) -> dict[str, Any]:
    """Ejecuta QA, resumen y generación sin publicar hechos generados."""
    assert modelos is not None
    if productos.empty:
        return {"question_answering": "sin contexto", "resumen": "sin contexto", "generacion": "no publicada"}
    contexto = "\n".join(
        f"Producto: {fila['Nombre Producto']}. Marca: {fila['Marca']}. "
        f"Tipo: {fila['Tipo Producto']}. Precio: ${fila['precio_num']:.2f}. "
        f"Descripción: {fila['Descripción y Especificaciones']}."
        for _, fila in productos.head(4).iterrows()
    )
    resultado: dict[str, Any] = {}
    try:
        qa = modelos.question_answering(question=pregunta, context=contexto[:3500])
        resultado["question_answering"] = {
            "answer": str(qa.get("answer", "")),
            "score": round(float(qa.get("score", 0.0)), 4),
        }
    except Exception as error:
        resultado["question_answering"] = f"no disponible: {type(error).__name__}"
    try:
        resumen = modelos.summarization(
            "Resume en español sin agregar datos: " + contexto[:2200],
            max_length=70,
            min_length=18,
            do_sample=False,
        )[0]["summary_text"]
        resultado["resumen"] = str(resumen)
    except Exception as error:
        resultado["resumen"] = f"no disponible: {type(error).__name__}"
    try:
        prompt = "Atención al cliente. Escribe una frase amable sin datos, precios ni promesas:"
        generado = modelos.text_generation(
            prompt,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=modelos.text_generation.tokenizer.eos_token_id,
        )[0]["generated_text"]
        resultado["generacion_no_publicada"] = str(generado)[len(prompt) :].strip()[:120]
    except Exception as error:
        resultado["generacion_no_publicada"] = f"no disponible: {type(error).__name__}"
    return resultado


def respuesta_final(estado: dict[str, Any], tono: str) -> tuple[str, pd.DataFrame, str]:
    productos, similitud = buscar_segun_estado(estado)
    tipo = estado.get("tipo_producto")
    marca = estado.get("marca")
    presupuesto = estado.get("presupuesto")
    criterios = [f"producto: {tipo}"]
    criterios.append(f"marca: {marca}" if marca else "marca: sin preferencia")
    criterios.append(
        f"presupuesto: hasta ${presupuesto:.2f}" if presupuesto is not None else "presupuesto: sin límite"
    )
    if estado.get("uso"):
        criterios.append(f"uso: {estado['uso']}")

    if productos.empty:
        base = candidatos_del_tipo(tipo)
        marcas = marcas_disponibles(tipo)
        rango = ""
        if not base.empty:
            rango = (
                f" Los precios registrados para este tipo van de "
                f"${base['precio_num'].min():.2f} a ${base['precio_num'].max():.2f}."
            )
        texto = (
            f"**Criterios usados:** {'; '.join(criterios)}.\n\n"
            f"No encontré productos que cumplan esa combinación. "
            f"Las marcas disponibles para **{tipo}** son: {', '.join(marcas)}.{rango} "
            "No ampliaré la búsqueda a otros artículos."
        )
        return texto, tabla_salida(productos), "No: puedes cambiar marca o presupuesto"

    intencion = estado.get("intencion")
    if intencion == "comparación de productos" and len(productos) < 2:
        opciones = alternativas_por_marca(tipo)
        unico = productos.iloc[0]
        texto = (
            f"**Criterios usados:** {'; '.join(criterios)}.\n\n"
            f"Solo encontré un producto que cumple los filtros: "
            f"**{unico['Nombre Producto']}** por **${unico['precio_num']:.2f}**. "
            "No puedo inventar un segundo producto para compararlo."
        )
        if opciones:
            texto += f" Para comparar dos, prueba con: {', '.join(opciones)} o sin preferencia de marca."
        return texto, tabla_salida(productos.head(1)), "No: faltan dos coincidencias comparables"

    limite = 3 if intencion in {"comparación de productos", "recomendación de producto"} else 10
    seleccionados = productos.head(limite)
    lineas = [f"**Criterios usados:** {'; '.join(criterios)}."]
    if tono == "Formal":
        lineas.append("Con gusto, estas son las coincidencias registradas en el catálogo:")
    else:
        lineas.append("Estas son las coincidencias reales del catálogo:")
    for _, fila in seleccionados.iterrows():
        lineas.append(
            f"- **{fila['Nombre Producto']}** — **${fila['precio_num']:.2f}**. "
            f"Marca: {fila['Marca']}. Disponibilidad registrada: {fila['Disponibilidad']}."
        )
    if intencion == "comparación de productos":
        lineas.append("La comparación está ordenada por precio cuando se solicita una opción económica.")

    requiere_humano = any(
        indicador in normalizar(estado.get("pregunta_inicial", ""))
        for indicador in ["stock hoy", "tienda de", "mi pedido", "llego", "danado"]
    )
    escalamiento = "No"
    if requiere_humano:
        escalamiento = "Sí: requiere verificar información operativa actual"
        lineas.append(
            "**Escalamiento:** confirma stock de tienda, pedidos o estado físico con Kywi "
            "al 1700 150 150 o WhatsApp +593 99 515 5150."
        )

    auxiliares = ejecutar_auxiliares(estado.get("pregunta_inicial", ""), seleccionados)
    print(
        json.dumps(
            {
                "intencion_conservada": intencion,
                "tipo_filtrado": tipo,
                "marca_filtrada": marca,
                "similitud_maxima": round(similitud, 4),
                "pipelines_auxiliares_no_publicados": auxiliares,
            },
            ensure_ascii=False,
        )
    )
    return "\n\n".join(lineas), tabla_salida(seleccionados), escalamiento


def respuesta_politica(estado: dict[str, Any]) -> tuple[str, pd.DataFrame, str]:
    intencion = estado["intencion"]
    titulo = POLITICAS_POR_INTENCION.get(intencion)
    if not titulo:
        texto = "No encontré una política específica para esa consulta."
    else:
        texto = extraer_politica(titulo)
    requiere = intencion == "reclamo" or "pedido" in normalizar(estado["pregunta_inicial"])
    if requiere:
        texto += (
            "\n\n**Escalamiento:** comunícate con Kywi al 1700 150 150 o "
            "WhatsApp +593 99 515 5150."
        )
    return texto, tabla_salida(pd.DataFrame()), "Sí" if requiere else "No"


def conversar(
    mensaje: str,
    historial: list[dict[str, str]] | None,
    estado: dict[str, Any] | None,
    tono: str,
) -> tuple[
    list[dict[str, str]],
    dict[str, Any],
    str,
    str,
    str,
    pd.DataFrame,
    str,
]:
    historial = list(historial or [])
    mensaje = str(mensaje or "").strip()
    if not mensaje:
        return (
            historial,
            estado or estado_nuevo(),
            "",
            "Sin clasificar",
            "Sin analizar",
            tabla_salida(pd.DataFrame()),
            "No",
        )

    historial.append({"role": "user", "content": mensaje})
    if modelos is None:
        iniciar_modelos()
    assert modelos is not None

    primer_turno = (
        not estado
        or not estado.get("pregunta_inicial")
        or estado.get("finalizado")
    )
    if primer_turno:
        estado = estado_nuevo()
        estado["pregunta_inicial"] = mensaje
        intencion, confianza = detectar_intencion(mensaje)
        resultado_sentimiento = modelos.sentiment(mensaje)[0]
        sentimiento, confianza_sentimiento, molesto = interpretar_sentimiento(
            resultado_sentimiento
        )
        estado.update(
            {
                "intencion": intencion,
                "confianza_intencion": confianza,
                "sentimiento": sentimiento,
                "confianza_sentimiento": confianza_sentimiento,
            }
        )

        if intencion not in INTENCIONES_PRODUCTO:
            respuesta, productos, escalamiento = respuesta_politica(estado)
            if molesto:
                respuesta = "Lamento la situación y entiendo tu molestia.\n\n" + respuesta
            historial.append({"role": "assistant", "content": respuesta})
            estado["finalizado"] = True
            return (
                historial,
                estado,
                "",
                f"{intencion} ({confianza:.1%})",
                f"{sentimiento} ({confianza_sentimiento:.1%})",
                productos,
                escalamiento,
            )

        tipo, confianza_tipo = detectar_tipo_producto(mensaje)
        estado["tipo_producto"] = tipo
        estado["confianza_tipo"] = confianza_tipo
        if tipo:
            marca_valida = detectar_marca_del_tipo(mensaje, tipo)
            marca_global = detectar_marca_global(mensaje)
            estado["marca"] = marca_valida
            estado["marca_definida"] = marca_valida is not None
            if marca_global and not marca_valida:
                estado["marca_definida"] = False
        presupuesto = detectar_presupuesto(mensaje)
        estado["presupuesto"] = presupuesto
        estado["presupuesto_definido"] = presupuesto is not None
        estado["uso"] = detectar_uso(mensaje)

        pregunta_siguiente = siguiente_pregunta(estado)
        if pregunta_siguiente:
            prefijo = f"Detecté la intención **{intencion}**."
            marca_global = detectar_marca_global(mensaje)
            if tipo and marca_global and not estado.get("marca"):
                prefijo += (
                    f" No encontré **{tipo}** de marca **{marca_global}** en el CSV."
                )
            respuesta = f"{prefijo}\n\n{pregunta_siguiente}"
            historial.append({"role": "assistant", "content": respuesta})
            return (
                historial,
                estado,
                "",
                f"{intencion} ({confianza:.1%})",
                f"{sentimiento} ({confianza_sentimiento:.1%})",
                tabla_salida(pd.DataFrame()),
                "No: recopilando preferencias",
            )
    else:
        valido, confirmacion = registrar_seguimiento(mensaje, estado)
        if not valido:
            historial.append({"role": "assistant", "content": confirmacion})
            return (
                historial,
                estado,
                "",
                f"{estado['intencion']} ({estado['confianza_intencion']:.1%})",
                f"{estado['sentimiento']} ({estado['confianza_sentimiento']:.1%})",
                tabla_salida(pd.DataFrame()),
                "No: recopilando preferencias",
            )
        pregunta_siguiente = siguiente_pregunta(estado)
        if pregunta_siguiente:
            respuesta = f"{confirmacion}\n\n{pregunta_siguiente}"
            historial.append({"role": "assistant", "content": respuesta})
            return (
                historial,
                estado,
                "",
                f"{estado['intencion']} ({estado['confianza_intencion']:.1%})",
                f"{estado['sentimiento']} ({estado['confianza_sentimiento']:.1%})",
                tabla_salida(pd.DataFrame()),
                "No: recopilando preferencias",
            )

    respuesta, productos, escalamiento = respuesta_final(estado, tono)
    historial.append({"role": "assistant", "content": respuesta})
    estado["finalizado"] = True
    return (
        historial,
        estado,
        "",
        f"{estado['intencion']} ({estado['confianza_intencion']:.1%})",
        f"{estado['sentimiento']} ({estado['confianza_sentimiento']:.1%})",
        productos,
        escalamiento,
    )


def reiniciar() -> tuple[
    list[dict[str, str]], dict[str, Any], str, str, str, pd.DataFrame, str
]:
    return (
        [],
        estado_nuevo(),
        "",
        "Sin clasificar",
        "Sin analizar",
        tabla_salida(pd.DataFrame()),
        "No",
    )


def copiar_ejemplo(ejemplo: str) -> str:
    return ejemplo or ""


def construir_interfaz() -> gr.Blocks:
    assert catalogo is not None
    css = """
    .gradio-container {max-width: 1200px !important;}
    .titulo {text-align:center; margin-bottom:0.2rem;}
    .nota {text-align:center; color:#4d5f55;}
    """
    with gr.Blocks(title="Asistente Kywi", css=css) as demo:
        gr.Markdown("# Asistente inteligente de Kywi", elem_classes="titulo")
        gr.Markdown(
            f"Conversación guiada sobre {len(catalogo):,} productos. "
            "Primero filtra el producto; luego ofrece solo marcas compatibles.",
            elem_classes="nota",
        )
        estado = gr.State(estado_nuevo())
        with gr.Row():
            with gr.Column(scale=4):
                ejemplos = gr.Dropdown(
                    choices=EJEMPLOS,
                    label="Caso preparado (opcional)",
                )
                mensaje = gr.Textbox(
                    label="Mensaje del cliente",
                    lines=3,
                    placeholder="Ej.: Compara dos hidrolavadoras económicas.",
                )
                tono = gr.Radio(
                    ["Automático", "Formal", "Cercano"],
                    value="Automático",
                    label="Tono",
                )
                with gr.Row():
                    enviar = gr.Button("Enviar mensaje", variant="primary")
                    nuevo = gr.Button("Nueva conversación")
            with gr.Column(scale=7):
                conversacion = gr.Chatbot(
                    label="Conversación", type="messages", height=500
                )
                with gr.Row():
                    intencion = gr.Textbox(label="Intención conservada")
                    sentimiento = gr.Textbox(label="Sentimiento")
                escalamiento = gr.Textbox(label="Escalamiento")
        productos = gr.Dataframe(label="Productos filtrados", interactive=False)

        ejemplos.change(copiar_ejemplo, inputs=ejemplos, outputs=mensaje)
        for evento in [enviar.click, mensaje.submit]:
            evento(
                conversar,
                inputs=[mensaje, conversacion, estado, tono],
                outputs=[
                    conversacion,
                    estado,
                    mensaje,
                    intencion,
                    sentimiento,
                    productos,
                    escalamiento,
                ],
            )
        nuevo.click(
            reiniciar,
            outputs=[
                conversacion,
                estado,
                mensaje,
                intencion,
                sentimiento,
                productos,
                escalamiento,
            ],
        )
    return demo


def main() -> None:
    cargar_catalogo()
    iniciar_modelos()
    demo = construir_interfaz()
    demo.queue().launch(share=True, debug=False)


if __name__ == "__main__":
    main()
