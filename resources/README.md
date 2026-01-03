# Componente B — JSON oficiales (sin hardcode)

Este paquete define, de forma explícita y versionada, todos los artefactos JSON necesarios para que el Componente B funcione de manera independiente del Componente A.

## Archivos
- config/config_b.json: parámetros globales del motor (umbrales, reglas de parada, política híbrida y restricciones).
- data/item_bank.json: banco de ítems con metadatos + parámetros IRT.
- data/bkt_parameters.json: parámetros BKT por skill.
- data/skill_map.json: taxonomía (topic → skills) y prerrequisitos.
- state/student_state.json: plantilla/estado persistente del estudiante.
- contracts/*.sample.json: ejemplos de mensajes A→B y B→A.
- schemas/*.schema.json: JSON Schema para validación estricta de contratos.

## Regla
No se permite hardcodear parámetros del modelo: cualquier umbral o parámetro se configura aquí.
