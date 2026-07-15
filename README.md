# Asistente interactivo de Kywi

Esta versión fue reconstruida desde el CSV entregado por el usuario. Conserva sus nueve columnas originales y agrega campos derivados para que el bot filtre de forma estricta.

## Archivos para Google Colab

- `Asistente_Kywi_Interactivo_Colab.ipynb`
- `app_kywi_interactivo.py`
- `catalogo_kywi_mejorado.csv`
- `politicas_kywi.txt`

El archivo `Catalogo_Kywi_mejorado.xlsx` permite revisar la estructura, la taxonomía y la calidad de los datos, pero no es necesario subirlo a Colab.

## Orden de filtrado

1. Intención inicial, que se conserva durante toda la conversación.
2. Tipo de producto.
3. Producto principal, para excluir accesorios no solicitados.
4. Marca disponible dentro del tipo filtrado.
5. Presupuesto.
6. Uso.

Si una combinación no existe, la aplicación lo indica y no amplía la búsqueda a otros artículos.

## Prueba crítica

Pregunta `Compara dos hidrolavadoras económicas` y responde `STANLEY`. La aplicación debe rechazar la marca porque no existe ninguna hidrolavadora principal STANLEY en el CSV. Las marcas disponibles son TRUPER, BOSCH, ELITE y KARCHER.

## Ejecución

1. Abre el notebook en Google Colab.
2. Selecciona una GPU T4.
3. Ejecuta las celdas en orden.
4. Cuando aparezca el selector, sube juntos los tres archivos indicados.
5. Abre el enlace de Gradio generado por la última celda.
