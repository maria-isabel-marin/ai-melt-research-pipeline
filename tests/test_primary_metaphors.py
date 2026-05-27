import pandas as pd

from ai_melt.primary_metaphors import (
    balanced_evaluation_sample,
    compare_approaches,
    consolidate_approaches,
    extract_expression,
    flatten_approach_results,
    prepare_work_dataframe,
    process_llm_approach,
)


def test_prepare_work_dataframe_adds_context() -> None:
    df_n0 = pd.DataFrame(
        {
            "ID_documento": ["DOC-1", "DOC-1"],
            "ID_oracion": ["S-000001", "S-000002"],
            "oracion_texto": ["La guerra dejó heridas.", "La paz se construye."],
            "pagina": [1, 1],
        }
    )
    df_work = prepare_work_dataframe(df_n0, sample_mode=False)
    assert df_work["contexto_ampliado"].iloc[0].endswith("||| La paz se construye.")


def test_extract_expression_uses_focus_window() -> None:
    expression = extract_expression(
        "La paz se construye lentamente en los territorios",
        "construye",
        10,
        18,
        window=2,
    )
    assert expression == "paz se construye lentamente en"


def test_process_and_flatten_llm_results() -> None:
    df_work = pd.DataFrame(
        {
            "ID_documento": ["DOC-1"],
            "ID_oracion": ["S-000001"],
            "oracion_texto": ["La paz se construye."],
            "contexto_ampliado": ["||| La paz se construye. |||"],
            "pagina": [1],
        }
    )

    def runner(_sentence: str, _context: str, _candidates: str):
        return (
            {
                "metaforas": [
                    {
                        "expresion_metaforica": "se construye",
                        "foco": "construye",
                        "dominio_fuente": "CONSTRUCCIÓN",
                        "dominio_meta": "PAZ",
                        "metafora_conceptual": "LA PAZ ES UNA CONSTRUCCIÓN",
                        "correspondencias_ontologicas": [
                            {
                                "elemento_fuente": "obra",
                                "elemento_meta": "paz",
                                "evidencia_textual": "construye",
                            }
                        ],
                    }
                ]
            },
            10,
            5,
        )

    nested, metrics = process_llm_approach(df_work, "openai", runner)
    df_metaphors, df_ont, df_epi = flatten_approach_results(nested, "openai")
    assert metrics["tokens_in"] == 10
    assert df_metaphors["ID_expresion"].iloc[0] == "M-OPE-00001"
    assert len(df_ont) == 1
    assert df_epi.empty


def test_consolidate_compare_and_evaluation_sample() -> None:
    openai = pd.DataFrame(
        {
            "ID_expresion": ["M-OPE-00001"],
            "ID_oracion": ["S-000001"],
            "foco": ["construye"],
            "enfoque": ["openai"],
            "dominio_fuente": ["CONSTRUCCIÓN"],
            "metafora_conceptual": ["LA PAZ ES UNA CONSTRUCCIÓN"],
        }
    )
    claude = pd.DataFrame(
        {
            "ID_expresion": ["M-CLA-00001"],
            "ID_oracion": ["S-000001"],
            "foco": ["construye"],
            "enfoque": ["claude"],
            "dominio_fuente": ["CONSTRUCCIÓN"],
            "metafora_conceptual": ["LA PAZ ES UNA CONSTRUCCIÓN"],
        }
    )
    results = {"openai": openai, "claude": claude}
    consolidated = consolidate_approaches(results)
    comparison, kappa = compare_approaches(results)
    evaluation = balanced_evaluation_sample(results, sample_per_approach=1)

    assert consolidated["confianza_cross_enfoques"].max() == 2
    assert comparison is not None and kappa is not None
    assert len(evaluation) == 2
