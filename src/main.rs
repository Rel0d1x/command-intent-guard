//! Command Intent Guard — помогает Астре отличить произнесённую *команду* от
//! *вопроса* о слове/фразе, которая лишь похожа на триггер команды
//! («поставь таймер» — команда, «что такое таймер» — вопрос).
//!
//! Это не хук, перехватывающий встроенный матчинг триггеров (в протоколе
//! плагинов такого хука нет — сопоставление голосовой фразы с командами
//! пользователя закрыто внутри демона). Вместо этого плагин добавляет tool,
//! который модель может вызвать сама, когда сказанное почти совпадает с
//! триггером известной команды, но не уверена, команда это или вопрос о нём.
//!
//! Дополняет `whisper-stt-plus` (в соседней папке), а не заменяет его: тот
//! правит ослышки ASR на уровне распознавания речи (снэппинг «поставь
//! тайфер» → «поставь таймер»), этот — различает НАМЕРЕНИЕ уже верно
//! распознанной фразы.

use astra_plugin_sdk::prelude::*;

#[astra::args]
#[derive(PluginConfig)]
#[serde(default)]
pub struct GuardConfig {
    /// Сходство (0-100), выше которого фраза считается командой даже без
    /// явных вопросительных или командных маркеров.
    similarity_threshold: u32,
    /// Дополнительные слова-признаки вопроса, в дополнение к встроенному списку.
    extra_question_words: Vec<String>,
    /// Дополнительные слова-признаки команды, в дополнение к встроенному списку.
    extra_command_words: Vec<String>,
}

impl Default for GuardConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 80,
            extra_question_words: Vec::new(),
            extra_command_words: Vec::new(),
        }
    }
}

/// Аргументы `classify_command_intent`.
#[astra::args]
pub struct ClassifyArgs {
    /// Точная фраза, которую произнёс или написал пользователь.
    utterance: String,
    /// Название команды, с которой Астра сравнивает фразу.
    command_name: String,
    /// Точная фраза-триггер этой команды, если отличается от названия.
    trigger_phrase: Option<String>,
}

const QUESTION_MARKERS: &[&str] = &[
    "что такое",
    "что значит",
    "что означает",
    "объясни",
    "поясни",
    "расскажи",
    "почему",
    "зачем",
    "что если",
    "разве",
    "неужели",
    "что",
    "как",
    "какой",
    "какая",
    "какое",
    "какие",
    "когда",
    "где",
    "кто",
    "чем",
    "ли",
    "what is",
    "what's",
    "what does",
    "why",
    "explain",
    "define",
    "how come",
];

const COMMAND_MARKERS: &[&str] = &[
    "поставь",
    "поставьте",
    "включи",
    "включите",
    "выключи",
    "выключите",
    "запусти",
    "запустите",
    "останови",
    "остановите",
    "напомни",
    "напомните",
    "открой",
    "откройте",
    "закрой",
    "закройте",
    "найди",
    "найдите",
    "покажи",
    "покажите",
    "добавь",
    "добавьте",
    "удали",
    "удалите",
    "установи",
    "установите",
    "выполни",
    "выполните",
    "скажи",
    "скажите",
    "повтори",
    "повторите",
    "отправь",
    "отправьте",
    "создай",
    "создайте",
    "запиши",
    "запишите",
    "turn on",
    "turn off",
    "set",
    "start",
    "stop",
    "remind",
    "open",
    "close",
    "find",
    "show",
    "add",
    "delete",
    "remove",
    "run",
    "play",
    "pause",
    "send",
    "create",
];

fn normalize(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (n, m) = (a.len(), b.len());
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr = vec![0usize; m + 1];
    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

fn similarity_pct(a: &str, b: &str) -> u32 {
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        return 100;
    }
    let dist = levenshtein(a, b);
    (100 * max_len.saturating_sub(dist) / max_len) as u32
}

/// Ищет маркер в нормализованном тексте: однословные — по точному совпадению
/// слова, многословные — по вхождению подстроки.
fn find_marker(text: &str, built_in: &[&str], extra: &[String]) -> Option<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let hits = |m: &str| -> bool {
        if m.contains(' ') {
            text.contains(m)
        } else {
            words.iter().any(|w| *w == m)
        }
    };
    built_in
        .iter()
        .find(|m| hits(m))
        .map(|m| m.to_string())
        .or_else(|| extra.iter().find(|m| hits(&m.to_lowercase())).cloned())
}

struct Verdict {
    intent: &'static str,
    confidence: f32,
    reasoning: String,
}

#[derive(Default)]
pub struct CommandIntentGuard {
    config: Config<GuardConfig>,
}

// ── логика классификации: обычный impl, макрос его не трогает ───────────────

impl CommandIntentGuard {
    fn classify(&self, a: &ClassifyArgs, cfg: &GuardConfig) -> Verdict {
        let utterance = normalize(&a.utterance);
        let trigger = normalize(a.trigger_phrase.as_deref().unwrap_or(&a.command_name));

        let mut score: i32 = 0;
        let mut notes: Vec<String> = Vec::new();

        if a.utterance.trim_end().ends_with('?') {
            score -= 4;
            notes.push("фраза заканчивается вопросительным знаком".into());
        }
        if let Some(w) = find_marker(&utterance, QUESTION_MARKERS, &cfg.extra_question_words) {
            score -= 3;
            notes.push(format!("найдено вопросительное слово «{w}»"));
        }
        if let Some(w) = find_marker(&utterance, COMMAND_MARKERS, &cfg.extra_command_words) {
            score += 3;
            notes.push(format!("найден командный глагол «{w}»"));
        }
        let sim = similarity_pct(&utterance, &trigger);
        if sim >= cfg.similarity_threshold {
            score += 1;
            notes.push(format!("высокое сходство с триггером ({sim}%)"));
        }

        let (intent, base_conf) = if score >= 3 {
            ("command", 0.65)
        } else if score <= -3 {
            ("question", 0.65)
        } else {
            ("ambiguous", 0.5)
        };
        let extra_conf = (score.unsigned_abs() as f32 - 3.0).max(0.0) * 0.07;
        let confidence = (base_conf + extra_conf).min(0.97);

        let reasoning = if notes.is_empty() {
            "Явных признаков не найдено — по формулировке решить нельзя, стоит переспросить пользователя."
                .to_string()
        } else {
            notes.join("; ")
        };

        Verdict { intent, confidence, reasoning }
    }
}

// ── что видит Астра ──────────────────────────────────────────────────────────

#[astra::plugin]
impl CommandIntentGuard {
    /// Проверь, что на самом деле имел в виду пользователь: команду или
    /// вопрос о похожем слове. Вызывай ПЕРЕД выполнением команды, если
    /// сказанная фраза почти совпадает с триггером команды, но могла быть
    /// вопросом об этом слове («поставь таймер» — команда, «что такое
    /// таймер» — вопрос). При intent="ambiguous" лучше переспросить
    /// пользователя, а не выполнять команду молча.
    #[tool]
    async fn classify_command_intent(&self, a: ClassifyArgs) -> Result<String, ToolError> {
        if a.utterance.trim().is_empty() {
            return Err(ToolError::BadArguments("utterance is empty".into()));
        }
        if a.command_name.trim().is_empty() {
            return Err(ToolError::BadArguments("command_name is empty".into()));
        }
        let cfg = self.config.load();
        let v = self.classify(&a, &cfg);
        Ok(json!({
            "intent": v.intent,
            "confidence": v.confidence,
            "reasoning": v.reasoning,
        })
        .to_string())
    }

    #[hook]
    async fn on_config(&self, _ctx: &PluginContext, config: GuardConfig) {
        self.config.store(config);
    }

    #[hook]
    async fn health_check(&self) -> (bool, String) {
        (true, "ok".into())
    }
}

astra::main!(CommandIntentGuard::default());
