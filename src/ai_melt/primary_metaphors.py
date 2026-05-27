"""Primary metaphor processing utilities for the AI-MELT research pipeline."""

# ruff: noqa: E501

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ai_melt.paths import project_path

SYSTEM_PROMPT = """Eres un lingüista cognitivo experto en la Teoría de la Metáfora Conceptual (Lakoff y Johnson, 1980) y en el procedimiento MIPVU (Steen et al., 2010) para la identificación de metáforas. Tu tarea es analizar oraciones del Informe Final de la Comisión de la Verdad de Colombia e identificar expresiones metafóricas.

## Procedimiento MIPVU que debes seguir:
1. Lee la oración completa y su contexto
2. Para cada unidad léxica de contenido (sustantivos, verbos, adjetivos, adverbios):
   a. Determina su SIGNIFICADO CONTEXTUAL: el sentido que la palabra adquiere en esta oración específica
   b. Determina su SIGNIFICADO BÁSICO: el sentido más concreto, más corporal, más preciso o históricamente anterior
   c. Si existe un CONTRASTE entre ambos significados, y el contextual puede explicarse mediante una transferencia desde el básico, la expresión es METAFÓRICA
3. Para cada expresión metafórica identificada, extrae todos los componentes analíticos

## Criterios importantes:
- Las metáforas convencionales (lexicalizadas) TAMBIÉN cuentan: "caer en la violencia", "construir la paz"
- Las personificaciones SON metáforas: "la violencia habla", "el país necesita"
- NO marques como metáfora: expresiones literales, metonimias puras, modismos sin base metafórica
- Si no hay metáforas en la oración, devuelve una lista vacía

## Formato de salida (JSON estricto):
Responde SOLO con un JSON válido, sin texto adicional, sin markdown, sin backticks.
"""

USER_PROMPT_TEMPLATE = """Analiza la siguiente oración y su contexto. Identifica TODAS las expresiones metafóricas.

CONTEXTO (oración anterior ||| oración actual ||| oración siguiente):
{contexto}

ORACIÓN A ANALIZAR:
{oracion}

{candidatos_info}

Para cada metáfora encontrada, devuelve un JSON con esta estructura exacta:
{{
  "metaforas": [
    {{
      "expresion_metaforica": "la frase metafórica tal como aparece en el texto",
      "foco": "la palabra que porta el sentido metafórico",
      "foco_lemma": "lema del foco",
      "foco_part_of_speech": "VERB|NOUN|ADJ|ADV",
      "significado_contextual": "sentido de la palabra en esta oración",
      "significado_basico": "sentido más concreto, corporal o históricamente anterior",
      "dominio_fuente": "DOMINIO FUENTE en mayúsculas (el concreto)",
      "dominio_meta": "DOMINIO META en mayúsculas (el abstracto)",
      "metafora_conceptual": "DOMINIO META ES DOMINIO FUENTE",
      "correspondencias_ontologicas": [
        {{
          "elemento_fuente": "entidad/propiedad en el dominio fuente",
          "elemento_meta": "entidad/propiedad en el dominio meta",
          "evidencia_textual": "fragmento del texto que sustenta este mapeo"
        }}
      ],
      "correspondencias_epistemicas": [
        {{
          "tipo_inferencia": "CAUSAL|TEMPORAL|CONDICIONAL|NORMATIVA|EVALUATIVA",
          "relacion_fuente": "conocimiento inferencial en el dominio fuente",
          "inferencia_meta": "conocimiento transferido al dominio meta",
          "evidencia_textual": "fragmento del texto"
        }}
      ]
    }}
  ]
}}

Si no hay metáforas en la oración, responde: {{"metaforas": []}}
"""

FEW_SHOT_EXAMPLES = [
    {
        "oracion": (
            "El conflicto armado ha dejado heridas profundas en el tejido social "
            "colombiano."
        ),
        "respuesta": {
            "metaforas": [
                {
                    "expresion_metaforica": "heridas profundas en el tejido social",
                    "foco": "heridas",
                    "foco_lemma": "herida",
                    "foco_part_of_speech": "NOUN",
                    "significado_contextual": (
                        "daños emocionales, traumas colectivos causados por el conflicto"
                    ),
                    "significado_basico": (
                        "lesión física en el cuerpo, corte o abertura en la piel"
                    ),
                    "dominio_fuente": "CUERPO FÍSICO",
                    "dominio_meta": "SOCIEDAD",
                    "metafora_conceptual": "LA SOCIEDAD ES UN CUERPO",
                    "correspondencias_ontologicas": [
                        {
                            "elemento_fuente": "herida en el cuerpo",
                            "elemento_meta": "trauma colectivo",
                            "evidencia_textual": "heridas profundas",
                        },
                        {
                            "elemento_fuente": "tejido corporal",
                            "elemento_meta": "estructura social",
                            "evidencia_textual": "tejido social",
                        },
                    ],
                    "correspondencias_epistemicas": [
                        {
                            "tipo_inferencia": "CAUSAL",
                            "relacion_fuente": (
                                "Las heridas causan dolor y debilitan el cuerpo"
                            ),
                            "inferencia_meta": (
                                "Los traumas causan sufrimiento y debilitan la sociedad"
                            ),
                            "evidencia_textual": (
                                "heridas profundas en el tejido social"
                            ),
                        },
                    ],
                }
            ]
        },
    }
]

LLMCallable = Callable[[str, str, str], tuple[dict[str, Any], int, int]]


def next_id(approach: str, counters: dict[str, int]) -> str:
    """Return the next expression id for an approach."""
    counters[approach] = counters.get(approach, 0) + 1
    return f"M-{approach[:3].upper()}-{counters[approach]:05d}"


def extract_expression(
    sentence: str,
    focus_token: str,
    focus_start: int,
    focus_end: int,
    window: int = 5,
) -> str:
    """Extract a word-window expression around a focus token."""
    words = sentence.split()
    char_pos = 0
    focus_word_idx = None
    for index, word in enumerate(words):
        if focus_start <= char_pos <= focus_end:
            focus_word_idx = index
            break
        char_pos += len(word) + 1
    if focus_word_idx is None:
        for index, word in enumerate(words):
            if focus_token.lower() in word.lower():
                focus_word_idx = index
                break
    if focus_word_idx is None:
        return focus_token
    start = max(0, focus_word_idx - window)
    end = min(len(words), focus_word_idx + window + 1)
    return " ".join(words[start:end])


def strip_json_fences(text: str) -> str:
    """Remove occasional markdown fences from LLM JSON output."""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    return re.sub(r"\s*```$", "", text).strip()


def build_llm_messages(
    sentence: str,
    context: str,
    candidates_info: str = "",
) -> list[dict[str, str]]:
    """Build chat messages matching the legacy MIPVU prompt."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in FEW_SHOT_EXAMPLES:
        messages.append(
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    oracion=example["oracion"],
                    contexto=f"||| {example['oracion']} |||",
                    candidatos_info="",
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(example["respuesta"], ensure_ascii=False),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                oracion=sentence, contexto=context, candidatos_info=candidates_info
            ),
        }
    )
    return messages


def make_openai_runner(llm_config: dict[str, Any]) -> LLMCallable:
    """Create an OpenAI chat-completions runner for the configured model."""
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("Install openai to run the OpenAI approach.") from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the OpenAI approach.")
    client = openai.OpenAI(api_key=api_key)

    def run(sentence: str, context: str, candidates_info: str = ""):
        messages = build_llm_messages(sentence, context, candidates_info)
        for attempt in range(int(llm_config.get("max_retries", 3))):
            try:
                response = client.chat.completions.create(
                    model=llm_config.get("openai_model", "gpt-4.1-mini"),
                    messages=messages,
                    temperature=float(llm_config.get("temperature", 0.1)),
                    max_tokens=int(llm_config.get("max_tokens", 4096)),
                    response_format={"type": "json_object"},
                )
                payload = json.loads(
                    strip_json_fences(response.choices[0].message.content)
                )
                return (
                    payload,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )
            except (json.JSONDecodeError, Exception):
                if attempt >= int(llm_config.get("max_retries", 3)) - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        return {"metaforas": []}, 0, 0

    return run


def make_claude_runner(llm_config: dict[str, Any]) -> LLMCallable:
    """Create a Claude runner for the configured model."""
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("Install anthropic to run the Claude approach.") from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for the Claude approach.")
    client = anthropic.Anthropic(api_key=api_key)

    def run(sentence: str, context: str, candidates_info: str = ""):
        messages = build_llm_messages(sentence, context, candidates_info)[1:]
        for attempt in range(int(llm_config.get("max_retries", 3))):
            try:
                response = client.messages.create(
                    model=llm_config.get("claude_model", "claude-sonnet-4-5"),
                    max_tokens=int(llm_config.get("max_tokens", 4096)),
                    temperature=float(llm_config.get("temperature", 0.1)),
                    system=[
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=messages,
                )
                payload = json.loads(strip_json_fences(response.content[0].text))
                return (
                    payload,
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                )
            except (json.JSONDecodeError, Exception):
                if attempt >= int(llm_config.get("max_retries", 3)) - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        return {"metaforas": []}, 0, 0

    return run


def prepare_work_dataframe(
    df_n0: pd.DataFrame,
    sample_mode: bool = True,
    sample_size: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """Select the configured N0 sample and add previous/current/next context."""
    if sample_mode:
        df_work = df_n0.sample(
            n=min(sample_size, len(df_n0)), random_state=random_state
        )
    else:
        df_work = df_n0.copy()
    df_work = df_work.sort_values(["ID_documento", "ID_oracion"]).reset_index(drop=True)

    contexts = []
    for index, row in df_work.iterrows():
        previous_sentence = (
            df_work.iloc[index - 1]["oracion_texto"]
            if index > 0
            and df_work.iloc[index - 1]["ID_documento"] == row["ID_documento"]
            else ""
        )
        next_sentence = (
            df_work.iloc[index + 1]["oracion_texto"]
            if index < len(df_work) - 1
            and df_work.iloc[index + 1]["ID_documento"] == row["ID_documento"]
            else ""
        )
        contexts.append(
            f"{previous_sentence} ||| {row['oracion_texto']} ||| {next_sentence}"
        )
    df_work["contexto_ampliado"] = contexts
    return df_work


def process_llm_approach(
    df_work: pd.DataFrame,
    approach: str,
    runner: LLMCallable,
    counters: dict[str, int] | None = None,
    rate_limit_pause_seconds: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one LLM approach over a prepared sentence table."""
    if counters is None:
        counters = {}
    results: list[dict[str, Any]] = []
    tokens_in = 0
    tokens_out = 0
    start = time.time()

    for _, row in df_work.iterrows():
        payload, token_in, token_out = runner(
            row["oracion_texto"], row["contexto_ampliado"], ""
        )
        tokens_in += token_in
        tokens_out += token_out
        for metaphor in payload.get("metaforas", []) or []:
            metaphor = metaphor.copy()
            metaphor["ID_expresion"] = next_id(approach, counters)
            metaphor["ID_oracion"] = row["ID_oracion"]
            metaphor["ID_documento"] = row["ID_documento"]
            metaphor["pagina"] = row.get("pagina", "")
            metaphor["contexto"] = row["oracion_texto"]
            metaphor["enfoque"] = approach
            metaphor["confianza_modelo"] = metaphor.get("confianza_modelo", 1.0)
            results.append(metaphor)
        if rate_limit_pause_seconds:
            time.sleep(rate_limit_pause_seconds)

    return results, {
        "seconds": time.time() - start,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }


def flatten_approach_results(
    results_list: list[dict[str, Any]], approach: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert nested metaphor results into flat metaphor/correspondence tables."""
    metaphors = []
    ontological = []
    epistemic = []
    for metaphor in results_list:
        expression_id = metaphor["ID_expresion"]
        metaphors.append(
            {
                "ID_expresion": expression_id,
                "ID_documento": metaphor.get("ID_documento", ""),
                "ID_oracion": metaphor.get("ID_oracion", ""),
                "pagina": metaphor.get("pagina", ""),
                "expresion_metaforica": metaphor.get("expresion_metaforica", ""),
                "contexto": metaphor.get("contexto", ""),
                "foco": metaphor.get("foco", ""),
                "foco_lemma": metaphor.get("foco_lemma", ""),
                "foco_part_of_speech": metaphor.get("foco_part_of_speech", ""),
                "significado_contextual": metaphor.get("significado_contextual", ""),
                "significado_basico": metaphor.get("significado_basico", ""),
                "dominio_fuente": metaphor.get("dominio_fuente", ""),
                "dominio_meta": metaphor.get("dominio_meta", ""),
                "metafora_conceptual": metaphor.get("metafora_conceptual", ""),
                "enfoque": approach,
                "confianza_modelo": metaphor.get("confianza_modelo", 1.0),
            }
        )
        for correspondence in metaphor.get("correspondencias_ontologicas", []) or []:
            ontological.append(
                {
                    "ID_correspondencia": f"CO-{expression_id}-{len(ontological) + 1:03d}",
                    "ID_expresion": expression_id,
                    "enfoque": approach,
                    "elemento_fuente": correspondence.get("elemento_fuente", ""),
                    "elemento_meta": correspondence.get("elemento_meta", ""),
                    "evidencia_textual": correspondence.get("evidencia_textual", ""),
                }
            )
        for correspondence in metaphor.get("correspondencias_epistemicas", []) or []:
            epistemic.append(
                {
                    "ID_correspondencia_ep": f"CE-{expression_id}-{len(epistemic) + 1:03d}",
                    "ID_expresion": expression_id,
                    "enfoque": approach,
                    "tipo_inferencia": correspondence.get("tipo_inferencia", ""),
                    "relacion_fuente": correspondence.get("relacion_fuente", ""),
                    "inferencia_meta": correspondence.get("inferencia_meta", ""),
                    "evidencia_textual": correspondence.get("evidencia_textual", ""),
                }
            )
    return pd.DataFrame(metaphors), pd.DataFrame(ontological), pd.DataFrame(epistemic)


def load_approach_results(
    approaches: list[str],
    metaphors_pattern: str,
) -> dict[str, pd.DataFrame]:
    """Load available approach-level metaphor CSVs."""
    available = {}
    for approach in approaches:
        path = project_path(metaphors_pattern.format(approach=approach))
        if path.exists():
            available[approach] = pd.read_csv(path)
    return available


def get_sentence_detection_vector(
    df: pd.DataFrame, all_sentences: list[str]
) -> list[int]:
    """Binary vector: one if the approach detected a metaphor in a sentence."""
    detected = set(df["ID_oracion"].unique())
    return [1 if sentence_id in detected else 0 for sentence_id in all_sentences]


def compare_approaches(
    results_by_approach: dict[str, pd.DataFrame],
    df_work: pd.DataFrame | None = None,
    costs: dict[str, float] | None = None,
    timings: dict[str, float] | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Calculate pairwise Cohen's kappa and overlap summaries."""
    approaches = list(results_by_approach)
    if len(approaches) < 2:
        return None, None
    all_sentence_ids: set[str] = set()
    for df in results_by_approach.values():
        all_sentence_ids.update(df["ID_oracion"].dropna().unique())
    if df_work is not None and "ID_oracion" in df_work:
        all_sentence_ids.update(df_work["ID_oracion"].dropna().unique())
    all_sentences = sorted(all_sentence_ids)

    kappa_matrix = pd.DataFrame(index=approaches, columns=approaches, dtype=float)
    rows = []
    for index_a, approach_a in enumerate(approaches):
        vector_a = get_sentence_detection_vector(
            results_by_approach[approach_a], all_sentences
        )
        for index_b, approach_b in enumerate(approaches):
            vector_b = get_sentence_detection_vector(
                results_by_approach[approach_b], all_sentences
            )
            kappa_matrix.loc[approach_a, approach_b] = (
                1.0 if index_a == index_b else cohen_kappa_score(vector_a, vector_b)
            )
            if index_a < index_b:
                shared_source_domains = len(
                    set(results_by_approach[approach_a]["dominio_fuente"].dropna())
                    & set(results_by_approach[approach_b]["dominio_fuente"].dropna())
                )
                shared_conceptual_metaphors = len(
                    set(results_by_approach[approach_a]["metafora_conceptual"].dropna())
                    & set(
                        results_by_approach[approach_b]["metafora_conceptual"].dropna()
                    )
                )
                rows.append(
                    {
                        "enfoque_A": approach_a,
                        "enfoque_B": approach_b,
                        "kappa_oracion": round(
                            float(kappa_matrix.loc[approach_a, approach_b]), 3
                        ),
                        "dominios_fuente_compartidos": shared_source_domains,
                        "metaforas_conceptuales_compartidas": shared_conceptual_metaphors,
                        "costo_A": round((costs or {}).get(approach_a, 0.0), 2),
                        "costo_B": round((costs or {}).get(approach_b, 0.0), 2),
                        "tiempo_A": round((timings or {}).get(approach_a, 0.0), 1),
                        "tiempo_B": round((timings or {}).get(approach_b, 0.0), 1),
                    }
                )
    return pd.DataFrame(rows), kappa_matrix


def consolidate_approaches(
    results_by_approach: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate approach results and add cross-approach focus confidence."""
    if not results_by_approach:
        return pd.DataFrame()
    df_consolidated = pd.concat(results_by_approach.values(), ignore_index=True)
    if df_consolidated.empty:
        return df_consolidated

    confidence_by_focus = (
        df_consolidated.groupby(["ID_oracion", "foco"])["enfoque"].nunique().to_dict()
    )

    def confidence(row: pd.Series) -> int:
        return confidence_by_focus.get((row["ID_oracion"], row.get("foco", "")), 1)

    df_consolidated["confianza_cross_enfoques"] = df_consolidated.apply(
        confidence, axis=1
    )
    return df_consolidated


def balanced_evaluation_sample(
    results_by_approach: dict[str, pd.DataFrame],
    sample_per_approach: int = 15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create the balanced human-evaluation sample from available approaches."""
    samples = []
    for _, df in results_by_approach.items():
        if not df.empty:
            samples.append(
                df.sample(
                    n=min(sample_per_approach, len(df)), random_state=random_state
                )
            )
    if not samples:
        return pd.DataFrame()
    df_eval = pd.concat(samples, ignore_index=True)
    columns = [
        "ID_expresion",
        "enfoque",
        "contexto",
        "expresion_metaforica",
        "foco",
        "foco_part_of_speech",
        "significado_contextual",
        "significado_basico",
        "dominio_fuente",
        "dominio_meta",
        "metafora_conceptual",
        "confianza_modelo",
    ]
    df_eval = df_eval[
        [column for column in columns if column in df_eval.columns]
    ].copy()
    df_eval["es_metafora_correcta"] = ""
    df_eval["componentes_correctos"] = ""
    df_eval["observaciones"] = ""
    return df_eval


def write_stage_01_tables(
    df_work: pd.DataFrame,
    results_by_approach: dict[str, pd.DataFrame],
    ontological_tables: list[pd.DataFrame],
    epistemic_tables: list[pd.DataFrame],
    comparison: pd.DataFrame | None,
    consolidated: pd.DataFrame,
    evaluation_sample: pd.DataFrame,
    outputs_config: dict[str, str],
) -> dict[str, Path]:
    """Write N1 CSV outputs."""
    written = {}
    for approach, df in results_by_approach.items():
        path = project_path(
            outputs_config["metaphors_pattern"].format(approach=approach)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        written[f"metaphors_{approach}"] = path

    sample_path = project_path(outputs_config["sentence_sample_csv"])
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    df_work.to_csv(sample_path, index=False, encoding="utf-8-sig")
    written["sentence_sample"] = sample_path

    if ontological_tables:
        path = project_path(outputs_config["ontological_correspondences_csv"])
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(ontological_tables, ignore_index=True).to_csv(
            path, index=False, encoding="utf-8-sig"
        )
        written["ontological_correspondences"] = path
    if epistemic_tables:
        path = project_path(outputs_config["epistemic_correspondences_csv"])
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(epistemic_tables, ignore_index=True).to_csv(
            path, index=False, encoding="utf-8-sig"
        )
        written["epistemic_correspondences"] = path
    if comparison is not None:
        path = project_path(outputs_config["approach_comparison_csv"])
        path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(path, index=False, encoding="utf-8-sig")
        written["approach_comparison"] = path
    if not consolidated.empty:
        path = project_path(outputs_config["primary_metaphors_csv"])
        path.parent.mkdir(parents=True, exist_ok=True)
        consolidated.to_csv(path, index=False, encoding="utf-8-sig")
        written["primary_metaphors"] = path
    if not evaluation_sample.empty:
        path = project_path(outputs_config["evaluation_sample_csv"])
        path.parent.mkdir(parents=True, exist_ok=True)
        evaluation_sample.to_csv(path, index=False, encoding="utf-8-sig")
        written["evaluation_sample"] = path
    return written


def cost_estimate(approach: str, tokens_in: int, tokens_out: int) -> float:
    """Preserve the legacy notebook's rough USD cost estimates."""
    if approach == "claude":
        return (tokens_in / 1e6 * 3) + (tokens_out / 1e6 * 15)
    if approach == "openai":
        return (tokens_in / 1e6 * 2.5) + (tokens_out / 1e6 * 10)
    return 0.0


def runner_for_approach(approach: str, llm_config: dict[str, Any]) -> LLMCallable:
    """Return the configured runner for supported LLM approaches."""
    if approach == "openai":
        return make_openai_runner(llm_config)
    if approach == "claude":
        return make_claude_runner(llm_config)
    raise NotImplementedError(
        f"Approach {approach!r} is configured but not implemented in the package yet."
    )


def run_stage_01(config: dict[str, Any]) -> pd.DataFrame:
    """Run primary metaphor extraction for configured approaches."""
    stage_config = config["stage_01"]
    df_n0 = pd.read_csv(project_path(stage_config["inputs"]["corpus_csv"]))
    sampling = stage_config.get("sampling", {})
    df_work = prepare_work_dataframe(
        df_n0,
        bool(sampling.get("sample_mode", True)),
        int(sampling.get("sample_size", 50)),
        int(sampling.get("random_state", 42)),
    )

    results_by_approach = {}
    ontological_tables = []
    epistemic_tables = []
    timings = {}
    costs = {}
    counters: dict[str, int] = {}
    llm_config = stage_config.get("llm", {})

    for approach in stage_config.get("approaches", {}).get("active", []):
        runner = runner_for_approach(approach, llm_config)
        nested_results, metrics = process_llm_approach(
            df_work,
            approach,
            runner,
            counters,
            float(llm_config.get("rate_limit_pause_seconds", 1.0)),
        )
        df_metaphors, df_ont, df_epi = flatten_approach_results(
            nested_results, approach
        )
        results_by_approach[approach] = df_metaphors
        if not df_ont.empty:
            ontological_tables.append(df_ont)
        if not df_epi.empty:
            epistemic_tables.append(df_epi)
        timings[approach] = metrics["seconds"]
        costs[approach] = cost_estimate(
            approach, metrics["tokens_in"], metrics["tokens_out"]
        )

    comparison, _ = compare_approaches(results_by_approach, df_work, costs, timings)
    consolidated = consolidate_approaches(results_by_approach)
    evaluation = balanced_evaluation_sample(
        results_by_approach,
        int(sampling.get("evaluation_sample_per_approach", 15)),
        int(sampling.get("random_state", 42)),
    )
    write_stage_01_tables(
        df_work,
        results_by_approach,
        ontological_tables,
        epistemic_tables,
        comparison,
        consolidated,
        evaluation,
        stage_config["outputs"],
    )
    return consolidated
