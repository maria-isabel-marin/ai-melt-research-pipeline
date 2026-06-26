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
from dotenv import load_dotenv
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
STAGE_01_APPROACHES = ["claude", "openai"]
STAGE_01_STEPS = [
    "config",
    "load-data",
    "design-prompt",
    "approach-a-claude",
    "approach-b-openai",
    "export-approach-results",
    "load-results",
    "compare-approaches",
    "consolidate-results",
    "human-evaluation-and-summary",
]


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

    api_key = require_api_key("openai")
    client = openai.OpenAI(api_key=api_key)

    def run(sentence: str, context: str, candidates_info: str = ""):
        messages = build_llm_messages(sentence, context, candidates_info)
        for attempt in range(int(llm_config.get("max_retries", 3))):
            try:
                response = client.chat.completions.create(
                    model=llm_config.get("model", "gpt-4.1-mini"),
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

    api_key = require_api_key("claude")
    client = anthropic.Anthropic(api_key=api_key)

    def run(sentence: str, context: str, candidates_info: str = ""):
        messages = build_llm_messages(sentence, context, candidates_info)[1:]
        for attempt in range(int(llm_config.get("max_retries", 3))):
            try:
                response = client.messages.create(
                    model=llm_config.get("model", "claude-sonnet-4-5"),
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
    successful_responses = 0
    failed_responses = 0
    start = time.time()

    for _, row in df_work.iterrows():
        try:
            payload, token_in, token_out = runner(
                row["oracion_texto"], row["contexto_ampliado"], ""
            )
            successful_responses += 1
        except Exception:
            failed_responses += 1
            continue
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
        "successful_responses": successful_responses,
        "failed_responses": failed_responses,
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


def load_environment() -> dict[str, bool]:
    """Load local environment variables and report credential presence safely."""
    load_dotenv(project_path(".env"), override=False)
    return {
        "claude": _usable_key(os.environ.get("ANTHROPIC_API_KEY")),
        "openai": _usable_key(os.environ.get("OPENAI_API_KEY")),
    }


def _usable_key(value: str | None) -> bool:
    return bool(value and value.strip() and "replace-with-real" not in value.lower())


def require_api_key(approach: str) -> str:
    """Return an API key or fail before an external request is attempted."""
    load_environment()
    env_name = {
        "claude": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }[approach]
    value = os.environ.get(env_name)
    if not _usable_key(value):
        raise RuntimeError(
            f"Missing {env_name}. Add it to .env or set it in the environment."
        )
    return str(value)


def _read_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_dataframe(
    df: pd.DataFrame, path: Path, write_csv: bool = False
) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    written = [path]
    if write_csv and path.suffix == ".parquet":
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        written.append(csv_path)
    return written


def _stage_path(stage: dict[str, Any], section: str, key: str, **values: str) -> Path:
    value = stage[section][key]
    formatted = str(value).format(**values) if values else value
    return project_path(formatted)


def _approach_config(stage: dict[str, Any], approach: str) -> dict[str, Any]:
    return {**stage.get("llm", {}), **stage["approaches"][approach]}


def _load_work(stage: dict[str, Any]) -> pd.DataFrame:
    return _read_dataframe(_stage_path(stage, "intermediate_outputs", "work_parquet"))


def _load_parsed(stage: dict[str, Any], approach: str) -> pd.DataFrame:
    return _read_dataframe(
        _stage_path(
            stage, "intermediate_outputs", "parsed_results_pattern", approach=approach
        )
    )


def _available_results(
    stage: dict[str, Any], exported: bool = True
) -> dict[str, pd.DataFrame]:
    section = "outputs" if exported else "intermediate_outputs"
    key = "metaphors_pattern" if exported else "parsed_results_pattern"
    results = {}
    for approach in STAGE_01_APPROACHES:
        path = _stage_path(stage, section, key, approach=approach)
        if path.exists():
            results[approach] = _read_dataframe(path)
    return results


def run_stage_01_step(
    config: dict[str, Any],
    step: str,
    *,
    write_csv: bool = False,
    sample_size: int | None = None,
    random_state: int | None = None,
    limit_sentences: int | None = None,
    document_id: str | None = None,
    page_number: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run one inspectable Stage 01 processing step."""
    stage = config["stage_01"]
    credentials = load_environment()
    if step == "config":
        path = _stage_path(stage, "outputs", "config_summary_json")
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "approaches": STAGE_01_APPROACHES,
            "input": str(_stage_path(stage, "inputs", "n0_corpus")),
            "credentials_present": credentials,
        }
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {"config_summary": path, "_summary": summary}

    if step == "load-data":
        input_path = _stage_path(stage, "inputs", "n0_corpus")
        df_n0 = _read_dataframe(input_path)
        original_rows = len(df_n0)
        if document_id is not None:
            df_n0 = df_n0[df_n0["ID_documento"] == document_id]
        if page_number is not None:
            df_n0 = df_n0[df_n0["pagina"] == page_number]
        sampling = stage.get("sampling", {})
        selected_size = sample_size or int(sampling.get("sample_size", 50))
        seed = random_state or int(sampling.get("random_state", 42))
        df_work = prepare_work_dataframe(
            df_n0,
            bool(sampling.get("sample_mode", True)),
            selected_size,
            seed,
        )
        if limit_sentences is not None:
            df_work = df_work.head(limit_sentences)
        outputs = _write_dataframe(
            df_work,
            _stage_path(stage, "intermediate_outputs", "work_parquet"),
            write_csv,
        )
        return {
            "outputs": outputs,
            "_summary": {
                "input": input_path,
                "rows_loaded": original_rows,
                "rows_retained": len(df_work),
                "documents": df_work["ID_documento"].nunique(),
                "chapters": (
                    df_work["capitulo"].nunique() if "capitulo" in df_work else 0
                ),
                "columns": list(df_work.columns),
            },
        }

    if step == "design-prompt":
        path = _stage_path(stage, "intermediate_outputs", "prompt_preview_txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        preview = SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TEMPLATE
        path.write_text(preview, encoding="utf-8")
        return {
            "prompt_preview": path,
            "_summary": {
                "name": stage["prompt"]["name"],
                "version": stage["prompt"]["version"],
                "characters": len(preview),
                "placeholders": ["contexto", "oracion", "candidatos_info"],
            },
        }

    approach_by_step = {
        "approach-a-claude": "claude",
        "approach-b-openai": "openai",
    }
    if step in approach_by_step:
        approach = approach_by_step[step]
        require_api_key(approach)
        raw_path = _stage_path(
            stage, "intermediate_outputs", "raw_results_pattern", approach=approach
        )
        parsed_path = _stage_path(
            stage,
            "intermediate_outputs",
            "parsed_results_pattern",
            approach=approach,
        )
        if parsed_path.exists() and not force:
            raise FileExistsError(
                f"{parsed_path} already exists. Use --force to rerun this API step."
            )
        work = _load_work(stage)
        runner = runner_for_approach(approach, _approach_config(stage, approach))
        nested, metrics = process_llm_approach(
            work,
            approach,
            runner,
            rate_limit_pause_seconds=float(
                stage["llm"].get("rate_limit_pause_seconds", 1.0)
            ),
        )
        if metrics["successful_responses"] == 0 and metrics["failed_responses"]:
            raise RuntimeError(
                f"{approach} returned no successful responses; no outputs were written."
            )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            json.dumps(nested, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        metaphors, ontological, epistemic = flatten_approach_results(nested, approach)
        outputs = [raw_path]
        outputs.extend(_write_dataframe(metaphors, parsed_path, write_csv))
        outputs.extend(
            _write_dataframe(
                ontological,
                _stage_path(
                    stage,
                    "intermediate_outputs",
                    "ontological_pattern",
                    approach=approach,
                ),
                write_csv,
            )
        )
        outputs.extend(
            _write_dataframe(
                epistemic,
                _stage_path(
                    stage,
                    "intermediate_outputs",
                    "epistemic_pattern",
                    approach=approach,
                ),
                write_csv,
            )
        )
        return {
            "outputs": outputs,
            "_summary": {
                "approach": approach,
                "model": stage["approaches"][approach]["model"],
                "sentences": len(work),
                "metaphors": len(metaphors),
                **metrics,
            },
        }

    if step == "export-approach-results":
        outputs = []
        row_counts = {}
        ontological_tables = []
        epistemic_tables = []
        for approach in STAGE_01_APPROACHES:
            parsed = _load_parsed(stage, approach)
            row_counts[approach] = len(parsed)
            outputs.extend(
                _write_dataframe(
                    parsed,
                    _stage_path(
                        stage, "outputs", "metaphors_pattern", approach=approach
                    ),
                    write_csv,
                )
            )
            ontological_tables.append(
                _read_dataframe(
                    _stage_path(
                        stage,
                        "intermediate_outputs",
                        "ontological_pattern",
                        approach=approach,
                    )
                )
            )
            epistemic_tables.append(
                _read_dataframe(
                    _stage_path(
                        stage,
                        "intermediate_outputs",
                        "epistemic_pattern",
                        approach=approach,
                    )
                )
            )
        outputs.extend(
            _write_dataframe(
                pd.concat(ontological_tables, ignore_index=True),
                _stage_path(stage, "outputs", "ontological_correspondences_parquet"),
                write_csv,
            )
        )
        outputs.extend(
            _write_dataframe(
                pd.concat(epistemic_tables, ignore_index=True),
                _stage_path(stage, "outputs", "epistemic_correspondences_parquet"),
                write_csv,
            )
        )
        return {"outputs": outputs, "_summary": {"rows": row_counts}}

    if step == "load-results":
        results = _available_results(stage)
        if not results:
            raise FileNotFoundError("No Claude or OpenAI approach result files found.")
        combined = pd.concat(results.values(), ignore_index=True)
        outputs = _write_dataframe(
            combined,
            _stage_path(stage, "intermediate_outputs", "loaded_results_parquet"),
            write_csv,
        )
        return {
            "outputs": outputs,
            "_summary": {
                "rows": {name: len(df) for name, df in results.items()},
                "columns": list(combined.columns),
                "missing_files": [
                    approach
                    for approach in STAGE_01_APPROACHES
                    if approach not in results
                ],
            },
        }

    if step == "compare-approaches":
        results = _available_results(stage)
        comparison, kappa = compare_approaches(results, _load_work(stage))
        if comparison is None or kappa is None:
            raise RuntimeError("Claude and OpenAI results are both required.")
        outputs = _write_dataframe(
            comparison,
            _stage_path(stage, "outputs", "approach_comparison_parquet"),
            write_csv,
        )
        csv_path = _stage_path(stage, "outputs", "approach_comparison_csv")
        comparison.to_csv(csv_path, index=False, encoding="utf-8-sig")
        kappa_path = _stage_path(stage, "outputs", "kappa_matrix_csv")
        kappa.to_csv(kappa_path, encoding="utf-8-sig")
        outputs.extend([csv_path, kappa_path])
        return {
            "outputs": outputs,
            "_summary": {
                "rows": {name: len(df) for name, df in results.items()},
                "matched_sentences": len(
                    set(results["claude"]["ID_oracion"])
                    & set(results["openai"]["ID_oracion"])
                ),
                "kappa": float(kappa.loc["claude", "openai"]),
            },
        }

    if step == "consolidate-results":
        results = _available_results(stage)
        consolidated = consolidate_approaches(results)
        outputs = _write_dataframe(
            consolidated,
            _stage_path(stage, "outputs", "primary_metaphors_parquet"),
            write_csv,
        )
        csv_path = _stage_path(stage, "outputs", "primary_metaphors_csv")
        consolidated.to_csv(csv_path, index=False, encoding="utf-8-sig")
        outputs.append(csv_path)
        return {
            "outputs": outputs,
            "_summary": {
                "rows": len(consolidated),
                "rule": stage["consolidation"]["rule"],
                "by_approach": consolidated["enfoque"].value_counts().to_dict(),
            },
        }

    if step == "human-evaluation-and-summary":
        results = _available_results(stage)
        human = stage["human_evaluation"]
        evaluation = balanced_evaluation_sample(
            results,
            int(human.get("sample_per_approach", 15)),
            random_state or int(human.get("random_state", 42)),
        )
        eval_path = _stage_path(stage, "outputs", "evaluation_sample_csv")
        evaluation.to_csv(eval_path, index=False, encoding="utf-8-sig")
        consolidated = _read_dataframe(
            _stage_path(stage, "outputs", "primary_metaphors_parquet")
        )
        work = _load_work(stage)
        represented = work[work["ID_oracion"].isin(consolidated["ID_oracion"])]
        summary = {
            "total_consolidated_rows": len(consolidated),
            "documents": consolidated["ID_documento"].nunique(),
            "chapters": (
                represented["capitulo"].nunique() if "capitulo" in represented else 0
            ),
            "approaches": STAGE_01_APPROACHES,
            "evaluation_sample_size": len(evaluation),
        }
        summary_path = _stage_path(stage, "outputs", "final_summary_json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {
            "outputs": [eval_path, summary_path],
            "_summary": summary,
        }
    raise ValueError(f"Unknown Stage 01 step: {step}")


def summarise_stage_01_result(step: str, result: dict[str, Any]) -> str:
    """Format compact Stage 01 processing diagnostics."""
    summary = result.get("_summary", {})
    lines = [f"Stage 01 processing step: {step}"]
    for key, value in summary.items():
        lines.append(f"{key.replace('_', ' ').title()}: {value}")
    outputs = result.get("outputs", [])
    outputs += [
        value
        for key, value in result.items()
        if key != "_summary" and isinstance(value, Path)
    ]
    if outputs:
        lines.append("Outputs written:")
        lines.extend(f"  - {path}" for path in outputs)
    return "\n".join(lines)


def run_stage_01(config: dict[str, Any]) -> pd.DataFrame:
    """Run the complete inspectable Stage 01 pipeline."""
    for step in STAGE_01_STEPS:
        run_stage_01_step(config, step)
    return pd.read_parquet(
        _stage_path(config["stage_01"], "outputs", "primary_metaphors_parquet")
    )
