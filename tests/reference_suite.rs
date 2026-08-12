//! Уровень 1 (см. dice-roller/tests/reference_suite.rs в AstraPlugins): хуки
//! гоняются в процессе через `Harness`, без демона и без сокета — здесь
//! проверяется именно логика классификации.

#[allow(dead_code, unused_imports)]
#[path = "../src/main.rs"]
mod plugin;

use astra_plugin_sdk::prelude::*;
use astra_plugin_sdk::testing::Harness;
use plugin::{ClassifyArgs, CommandIntentGuard};

fn guard() -> Harness<CommandIntentGuard> {
    Harness::new(CommandIntentGuard::default())
}

fn intent_of(out: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(out).unwrap();
    v["intent"].as_str().unwrap().to_string()
}

#[tokio::test]
async fn an_imperative_phrase_is_a_command() {
    let h = guard().start().await.unwrap();
    let out = h
        .call_tool(
            "classify_command_intent",
            json!({
                "utterance": "поставь таймер на пять минут",
                "command_name": "Таймер",
                "trigger_phrase": "таймер",
            }),
        )
        .await
        .unwrap();
    assert_eq!(intent_of(&out), "command", "{out}");
}

#[tokio::test]
async fn a_meta_question_about_the_trigger_word_is_not_a_command() {
    let h = guard().start().await.unwrap();
    let out = h
        .call_tool(
            "classify_command_intent",
            json!({
                "utterance": "что такое таймер",
                "command_name": "Таймер",
                "trigger_phrase": "таймер",
            }),
        )
        .await
        .unwrap();
    assert_eq!(intent_of(&out), "question", "{out}");
}

#[tokio::test]
async fn the_bare_trigger_word_alone_is_ambiguous() {
    let h = guard().start().await.unwrap();
    let out = h
        .call_tool(
            "classify_command_intent",
            json!({ "utterance": "таймер", "command_name": "Таймер" }),
        )
        .await
        .unwrap();
    assert_eq!(intent_of(&out), "ambiguous", "{out}");
}

/// Даже когда есть и командный глагол, и высокое сходство с триггером,
/// финальный «?» тянет счёт назад к неопределённости, а не к «command».
#[tokio::test]
async fn a_trailing_question_mark_pulls_an_imperative_phrase_back_to_ambiguous() {
    let h = guard().start().await.unwrap();
    let out = h
        .call_tool(
            "classify_command_intent",
            json!({
                "utterance": "выключи свет?",
                "command_name": "Свет",
                "trigger_phrase": "выключи свет",
            }),
        )
        .await
        .unwrap();
    assert_eq!(intent_of(&out), "ambiguous", "{out}");
}

#[tokio::test]
async fn extra_config_words_extend_the_built_in_lists() {
    let h = guard()
        .with_config(json!({ "extra_command_words": ["врубай"] }))
        .start()
        .await
        .unwrap();
    let out = h
        .call_tool(
            "classify_command_intent",
            json!({
                "utterance": "врубай таймер",
                "command_name": "Таймер",
                "trigger_phrase": "таймер",
            }),
        )
        .await
        .unwrap();
    assert_eq!(intent_of(&out), "command", "{out}");
}

/// Свежая установка получает `{}` — контейнерный `#[serde(default)]` и
/// `Default for GuardConfig` должны дать порог 80%, а не 0.
#[tokio::test]
async fn a_fresh_install_gets_the_documented_defaults() {
    let h = Harness::new(CommandIntentGuard::default())
        .with_config_json("{}")
        .start()
        .await
        .unwrap();
    let out = h
        .call_tool(
            "classify_command_intent",
            json!({ "utterance": "таймер", "command_name": "Таймер" }),
        )
        .await
        .unwrap();
    // 100% сходство >= порога по умолчанию (80) — сигнал засчитывается.
    assert!(out.contains("высокое сходство"), "{out}");
}

#[tokio::test]
async fn empty_utterance_is_bad_arguments_and_not_a_crash() {
    let h = guard().start().await.unwrap();
    let err = h
        .call_tool(
            "classify_command_intent",
            json!({ "utterance": "", "command_name": "Таймер" }),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, ToolError::BadArguments(_)), "{err:?}");

    let err = h.call_tool("no_such_tool", json!({})).await.unwrap_err();
    assert!(matches!(err, ToolError::NotFound(_)), "{err:?}");
}

#[tokio::test]
async fn the_tool_schema_matches_the_argument_type() {
    let h = guard().start().await.unwrap();
    let names: Vec<String> = h.tools().await.into_iter().map(|t| t.name).collect();
    assert_eq!(names, ["classify_command_intent"]);
    h.assert_schema_matches::<ClassifyArgs>("classify_command_intent").await;
}
