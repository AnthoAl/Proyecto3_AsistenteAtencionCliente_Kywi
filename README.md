# Asistente interactivo de Kywi

Construcción de un asistente capaz de comprender la intención del cliente, mantener el contexto de la conversación y solicitar los datos faltantes antes de recomendar un producto para la compañía Kywi. El asistente se basa en un catálogo de productos y políticas de la empresa, y utiliza un modelo de lenguaje para interactuar con el usuario.

## Archivos del Proyecto (Entorno Kaggle)

- `Proyecto3_ AsistenteAtencionCliente_Kywi.ipynb`
- `app_kywi_interactivo.py`
- `catalogo_kywi_mejorado.csv`
- `politicas_kywi.txt`

## Orden de filtrado

1. Intención inicial, que se conserva durante toda la conversación.
2. Tipo de producto.
3. Producto principal, para excluir accesorios no solicitados.
4. Marca disponible dentro del tipo filtrado.
5. Presupuesto.
6. Uso.

Si una combinación no existe, la aplicación lo indica y no amplía la búsqueda a otros artículos.

## Ejecución en Kaggle

Para facilitar la revisión y despliegue de la aplicación, los archivos fuente están enlazados a un repositorio de GitHub. El notebook se encarga de prepararlos de forma autónoma, por lo que **no es necesario subir ningún archivo manualmente**.

1. Abrir el notebook del proyecto en Kaggle.
2. En el panel superior en el apartado Configuraciones, verificar que el acelerador gráfico sea **GPU T4 x2** (o similar) en la sección *Accelerator*.
3. Verificar en ese mismo panel que el acceso a **Internet** esté encendido.
4. Ejecutar las celdas secuencialmente. La celda inicial descargará automáticamente `app_kywi_interactivo.py`, `catalogo_kywi_mejorado.csv` y `politicas_kywi.txt` desde GitHub hacia el directorio de trabajo (`/kaggle/working/`).
5. Al finalizar la ejecución de la última celda, abre el enlace público de Gradio (`Running on public URL: https://...`) para interactuar con la interfaz del asistente.