import pandas as pd

from ai_melt.primary_metaphors import (
    STAGE_01_APPROACHES,
    balanced_evaluation_sample,
    compare_approaches,
    consolidate_approaches,
    extract_expression,
    flatten_approach_results,
    prepare_work_dataframe,
    process_llm_approach,
    require_api_key,
    run_stage_01_step,
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


def test_stage_01_supports_only_claude_and_openai() -> None:
    assert STAGE_01_APPROACHES == ["claude", "openai"]


def test_missing_api_keys_fail_clearly(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "ai_melt.primary_metaphors.load_dotenv", lambda *args, **kwargs: False
    )

    for approach, env_name in [
        ("claude", "ANTHROPIC_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
    ]:
        try:
            require_api_key(approach)
        except RuntimeError as exc:
            assert env_name in str(exc)
            assert ".env" in str(exc)
        else:
            raise AssertionError("Expected missing key error")


def test_stage_01_load_data_and_prompt_steps(tmp_path) -> None:
    input_path = tmp_path / "n0.parquet"
    work_path = tmp_path / "work.parquet"
    prompt_path = tmp_path / "prompt.txt"
    pd.DataFrame(
        {
            "ID_documento": ["DOC-1", "DOC-1"],
            "ID_oracion": ["S-1", "S-2"],
            "oracion_texto": ["La paz se construye.", "La guerra deja heridas."],
            "pagina": [1, 1],
            "capitulo": ["C1", "C1"],
        }
    ).to_parquet(input_path, index=False)
    config = {
        "stage_01": {
            "inputs": {"n0_corpus": input_path},
            "prompt": {"name": "mipvu", "version": "1"},
            "approaches": {
                "claude": {"enabled": True, "model": "claude"},
                "openai": {"enabled": True, "model": "openai"},
            },
            "llm": {},
            "sampling": {
                "sample_mode": False,
                "sample_size": 50,
                "random_state": 42,
            },
            "intermediate_outputs": {
                "work_parquet": work_path,
                "prompt_preview_txt": prompt_path,
            },
        }
    }

    loaded = run_stage_01_step(config, "load-data", write_csv=True)
    prompt = run_stage_01_step(config, "design-prompt")

    assert loaded["_summary"]["rows_retained"] == 2
    assert work_path.exists()
    assert work_path.with_suffix(".csv").exists()
    assert prompt["prompt_preview"].exists()
